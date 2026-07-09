% Generate golden reference values for the penny-shaped crack (sill) from the
% original Fialko, Khazan & Simons (2001) MATLAB code.
%
%   Reference: Fialko, Y., Khazan, Y., Simons, M. (2001), Deformation due to a
%   pressurized horizontal circular crack in an elastic half-space, with
%   applications to volcano geodesy, GJI 146(1), 181-190.
%
% Fialko's penny-crack code is NOT redistributed in this repo (it carries no
% explicit redistribution license). Download the ORIGINAL into a local `penny/`
% subdirectory next to this script before running -- e.g.:
%   wget http://igppweb.ucsd.edu/~fialko/Assets/Software/penny.tar.gz
% Use the original, NOT the GeodMod mirror
% (https://github.com/falkamelung/GeodMod/tree/master/deformation_sources/penny):
% the two use different array-shape conventions (see below). (`penny/` is
% git-ignored.) Then:
%
% Run from this directory:  matlab -batch "run('gen_fialko.m')"
% Writes ../data/fialko_golden.json
%
% IMPORTANT -- scalar-vs-vectorized Q, and the GeodMod intgr.m bug:
%   The original penny code's Q(h,t,r,n) takes a *scalar* radius r (with t the
%   node vector), and its intgr.m already carries the correct per-radius Uz line
%       Uz = sum(Wt.*( fi.*(Q1+h*Q2) + psi.*(Q1./t - Q3) ))       % CORRECT
%   The GeodMod mirror instead vectorizes Q over the radii and its "speedup" Uz
%   line is WRONG (it factors `fi` over the psi terms):
%       Uz = sum(Wt2.*fi2.*(Qf1 + h*Qf2 + psi2.*Qf1./tt - Qf3))   % WRONG
%   Radial (Ur) is unaffected. torchdeform's penny.py implements the correct
%   formula, so we recompute Uz with the per-radius loop below (matching the
%   original scalar Q) rather than trusting intgr's Uz.
%
% Conventions match torchdeform PennySource: the dimensionless solution depends
% only on h = depth/radius; nu and mu enter purely through the scale factor
% Pf = 2(1-nu)*a*P/mu, and Uz_phys = -Uz_dimless*Pf, Ur_phys = Ur_dimless*Pf.
here = fileparts(mfilename('fullpath'));
penny_dir = fullfile(here,'penny');
if ~isfile(fullfile(penny_dir,'fredholm.m'))
    error(['Fialko penny code not found in %s\n' ...
           'Download it there first (see the header of this script).'], penny_dir);
end
addpath(penny_dir);
global NumLegendreTerms %#ok<GVMIS>

nu  = 0.25;
mu  = 3.0e10;
a   = 1000.0;      % crack radius (m)
P   = 1.0e6;       % excess pressure (Pa)
Pf  = 2.0*(1.0-nu)*a*P/mu;

nis = 2;           % sub-intervals (matches PennySource default)
eps = 1e-10;       % tight Fredholm convergence

r  = [0.001, 0.3, 0.6, 0.9, 1.2, 1.8, 2.5];   % dimensionless radius r/a
hs = [0.8, 1.5, 3.0];                          % dimensionless depth h = depth/a

out = struct();
out.meta = struct('nu',nu,'mu',mu,'a',a,'P',P,'Pf',Pf,'nis',nis,'r',r);

cases = {};
for k = 1:numel(hs)
    h = hs(k);
    [fi,psi,t,Wt] = fredholm(h, nis, eps);
    [~,Ur] = intgr(r, fi, psi, h, Wt, t);    % Ur is correct in intgr.m

    % Uz: use the ORIGINAL Fialko loop formula (see header note). The original
    % penny code's Q(h,t,r,n) takes a *scalar* radius r with the node vector t,
    % so evaluate per radius (this is intgr.m's own correct Uz line, recomputed
    % here so the result is independent of which intgr.m variant is installed).
    Uz = zeros(size(r));
    for j = 1:numel(r)
        rj = r(j);
        Q1 = Q(h, t, rj, 1);
        Q2 = Q(h, t, rj, 2);
        Q3 = Q(h, t, rj, 3);
        Uz(j) = sum(Wt.*( fi.*(Q1 + h*Q2) + psi.*(Q1./t - Q3) ));
    end

    cases{end+1} = struct('h',h,'depth',h*a, ...
                          'uz_dimless',Uz(:)','ur_dimless',Ur(:)', ...
                          'uz',-Uz(:)'*Pf,'ur',Ur(:)'*Pf); %#ok<SAGROW>
end
out.cases = cases;

outfile = fullfile(here,'..','data','fialko_golden.json');
fid = fopen(outfile,'w');
fwrite(fid, jsonencode(out, 'PrettyPrint', true));
fclose(fid);
fprintf('wrote %s\n', outfile);
