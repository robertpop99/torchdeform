% Generate golden reference values for the penny-shaped crack (sill) from the
% original Fialko, Khazan & Simons (2001) MATLAB code.
%
%   Reference: Fialko, Y., Khazan, Y., Simons, M. (2001), Deformation due to a
%   pressurized horizontal circular crack in an elastic half-space, with
%   applications to volcano geodesy, GJI 146(1), 181-190.
%
% Fialko's penny-crack code is NOT redistributed in this repo (it carries no
% explicit redistribution license). Download it yourself into a local `penny/`
% subdirectory next to this script before running -- e.g. from the GeodMod
% mirror:
%   https://github.com/falkamelung/GeodMod/tree/master/deformation_sources/penny
% The needed files are: Q.m Qr.m RtWt.m fpkernel.m fred.m fredholm.m intgr.m
% (`penny/` is git-ignored). Then:
%
% Run from this directory:  matlab -batch "run('gen_fialko.m')"
% Writes ../data/fialko_golden.json
%
% IMPORTANT -- bug in Fialko's intgr.m (as mirrored by GeodMod):
%   The active (vectorized "speedup") Uz line in penny/intgr.m is WRONG: it
%   factors `fi` over the psi terms,
%       Uz = sum(Wt2.*fi2.*(Qf1 + h*Qf2 + psi2.*Qf1./tt - Qf3))
%   The correct expression (the commented original loop in the same file) is
%       Uz = sum(Wt.*( fi.*(Q1+h*Q2) + psi.*(Q1./t - Q3) ))
%   Radial (Ur) is unaffected. torchdeform's penny.py implements the correct
%   formula, so we recompute Uz correctly below rather than trusting intgr's Uz.
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

    % Uz: use the ORIGINAL Fialko loop formula (see header note).
    rr  = repmat(reshape(r,numel(r),1), size(t));
    tt  = repmat(t,  numel(r),1);
    Wt2 = repmat(Wt, numel(r),1);
    fi2 = repmat(fi, numel(r),1);
    psi2= repmat(psi,numel(r),1);
    Qf  = Q(h,tt,rr,1:8);
    Uz  = sum(Wt2.*( fi2.*(Qf(:,:,1) + h*Qf(:,:,2)) ...
                   + psi2.*(Qf(:,:,1)./tt - Qf(:,:,3)) ), 2);
    Uz  = reshape(Uz, size(r));

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
