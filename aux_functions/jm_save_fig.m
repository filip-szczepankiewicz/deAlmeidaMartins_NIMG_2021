function fig = jm_save_fig(papersize, fig_fn, ext, res)
% function fig = jm_save_fig(papersize, fig_fn)
% Select res = 'Inf' for vectorial figures

if nargin < 2
    work_dir = pwd;
    fig_fn = fullfile(work_dir, 'Figure');
end
if (nargin < 3), ext = '-dpng'; end
if (nargin < 4), res = '-r300'; end

fig                = gcf;
fig.InvertHardcopy = 'off';
fig.Color          = [1 1 1];

if strcmp(res, 'Inf')
    res          = [];
    fig.Color    = 'none';
    fig.Renderer = 'Painters';
end

msf_mkdir(fileparts(fig_fn));
set(fig, 'PaperUnits','centimeters','PaperPosition', [0 0 papersize],'PaperSize', papersize);
print(fig_fn, ext, res)