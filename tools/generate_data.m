function generate_data()
% generate_data  Run all data generation scripts in the repo's data folder
%
% This function is location-aware: it finds the project root by walking up
% from the file location until it finds a folder containing a 'data'
% subfolder (or README.md). That makes it safe to move this file into
% the `tools/` folder without depending on the present working directory.

% Determine the directory of this file. For functions mfilename('fullpath')
% returns the full path; if empty, fall back to pwd.
this_file = mfilename('fullpath');
if isempty(this_file)
    start_dir = pwd;
else
    start_dir = fileparts(this_file);
end

% Walk up until we find a directory that contains a 'data' folder (or README.md)
project_root = start_dir;
max_up = 10;
found = false;
for i = 1:max_up
    if exist(fullfile(project_root, 'data'), 'dir') || exist(fullfile(project_root, 'README.md'), 'file')
        found = true;
        break;
    end
    parent = fileparts(project_root);
    if isempty(parent) || strcmp(parent, project_root)
        break;
    end
    project_root = parent;
end

if ~found
    error('generate_data:ProjectRootNotFound', ...
        'Could not locate project root containing a ''data'' folder. Call generate_data from within the repository or update this function.');
end

data_folder = fullfile(project_root, 'data');
subfolders = dir(data_folder);
subfolders = subfolders([subfolders.isdir] & ~startsWith({subfolders.name}, '.'));

for k = 1:length(subfolders)
    subfolder_path = fullfile(data_folder, subfolders(k).name);
    % Find all .m files starting with 'generate_' in the subfolder
    m_files = dir(fullfile(subfolder_path, 'generate_*.m'));
    for j = 1:length(m_files)
        script_path = fullfile(subfolder_path, m_files(j).name);
        fprintf('Running %s\n', script_path);
        run(script_path);
    end
end

fprintf('All data generation scripts executed.\n');
end
