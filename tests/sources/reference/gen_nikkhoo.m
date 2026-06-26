% Generate golden reference values for pCDM / CDM / pECM from the original
% Nikkhoo et al. (2017) MATLAB codes (vendored under reference/nikkhoo/).
%
%   Reference: Nikkhoo, M., Walter, T. R., Lundgren, P. R., Prats-Iraola, P.
%   (2017), Compound dislocation models (CDMs) for volcano deformation
%   analyses, GJI 208(2), 877-894.
%
% Run from this directory:  matlab -batch "run('gen_nikkhoo.m')"
% Writes ../data/nikkhoo_golden.json
%
% Material constants and parameter conventions match the torchdeform defaults
% (PCDMSource / CDMSource / PECMSource): nu = 0.25, mu = 3e10 Pa; rotation
% angles in degrees here (the Python API takes radians); CDM/pECM semi-axes
% (the reference doubles them internally); potencies in m^3.
here = fileparts(mfilename('fullpath'));
addpath(fullfile(here,'nikkhoo'));

nu     = 0.25;
mu     = 3.0e10;
lambda = 2*mu*nu/(1-2*nu);   % = 3e10

% fixed observation points (East x, North y), metres
X = [1000,    0, -2000, 1200,  -800, 3000];
Y = [   0, 1500,   500,-1800, -1200, 2500];

% common source location / depth (m)
X0 = 500; Y0 = -250; depth = 3000;

out = struct();
out.meta = struct('nu',nu,'mu',mu,'lambda',lambda, ...
                  'X0',X0,'Y0',Y0,'depth',depth,'X',X,'Y',Y);

% ---- pCDM: [omega(deg)  DVx DVy DVz(m^3)] ----
pcdm_sets = {
    struct('name','generic', 'om',[ 20 -10 30], 'dv',[1.4e6 1.2e6 0.8e6]);
    struct('name','isotropic','om',[  0   0  0], 'dv',[1.0e6 1.0e6 1.0e6]);
};
pcdm = {};
for i=1:numel(pcdm_sets)
    s = pcdm_sets{i};
    [ue,un,uv] = pCDM(X,Y,X0,Y0,depth, s.om(1),s.om(2),s.om(3), ...
                      s.dv(1),s.dv(2),s.dv(3), nu);
    pcdm{end+1} = struct('name',s.name,'omega',s.om,'dv',s.dv, ...
                         'ue',ue(:)','un',un(:)','uv',uv(:)'); %#ok<*SAGROW>
end
out.pcdm = pcdm;

% ---- CDM (finite): [omega(deg)  ax ay az(m, semi-axes)  opening(m)] ----
cdm_sets = {
    struct('name','generic',    'om',[15 -12 25], 'a',[300 250 180],'op',0.6);
    struct('name','axisaligned','om',[ 0   0  0], 'a',[400 300 200],'op',0.5);
};
cdm = {};
for i=1:numel(cdm_sets)
    s = cdm_sets{i};
    [ue,un,uv,DV] = CDM(X,Y,X0,Y0,depth, s.om(1),s.om(2),s.om(3), ...
                        s.a(1),s.a(2),s.a(3), s.op, nu);
    cdm{end+1} = struct('name',s.name,'omega',s.om,'a',s.a,'opening',s.op, ...
                        'DV',DV,'ue',ue(:)','un',un(:)','uv',uv(:)');
end
out.cdm = cdm;

% ---- pECM: [omega(deg)  ax ay az(m, semi-axes)  p(Pa)] ----
pecm_sets = {
    struct('name','triaxial','om',[10 20 -15], 'a',[350 250 150],'p',2.0e6);
    struct('name','aligned', 'om',[ 0  0   0], 'a',[300 200 150],'p',1.0e6);
};
pecm = {};
for i=1:numel(pecm_sets)
    s = pecm_sets{i};
    [ue,un,uv,dV,DV] = pECM(X,Y,X0,Y0,depth, s.om(1),s.om(2),s.om(3), ...
                            s.a(1),s.a(2),s.a(3), s.p, mu, lambda);
    pecm{end+1} = struct('name',s.name,'omega',s.om,'a',s.a,'p',s.p, ...
                         'dV',dV,'ue',ue(:)','un',un(:)','uv',uv(:)');
end
out.pecm = pecm;

outfile = fullfile(here,'..','data','nikkhoo_golden.json');
fid = fopen(outfile,'w');
fwrite(fid, jsonencode(out, 'PrettyPrint', true));
fclose(fid);
fprintf('wrote %s\n', outfile);
