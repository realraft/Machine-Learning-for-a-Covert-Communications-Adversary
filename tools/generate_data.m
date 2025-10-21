function generate_data()
%GENERATE_DATA Run all data generation scripts in the repository.

%% Locate project root
scriptDir = fileparts(mfilename('fullpath'));
if isempty(scriptDir)
    scriptDir = pwd;
end

projectRoot = scriptDir;
maxDepth = 10;
hasMarker = isfolder(fullfile(projectRoot, 'data')) || isfile(fullfile(projectRoot, 'README.md'));

for depth = 1:maxDepth
    if hasMarker
        break;
    end

    parent = fileparts(projectRoot);
    if strcmp(parent, projectRoot)
        projectRoot = '';
        break;
    end

    projectRoot = parent;
    hasMarker = isfolder(fullfile(projectRoot, 'data')) || isfile(fullfile(projectRoot, 'README.md'));
end

if isempty(projectRoot) || ~hasMarker
    fprintf('Project root not located. No scripts executed.\n');
    return;
end

dataFolder = fullfile(projectRoot, 'data');
if ~isfolder(dataFolder)
    fprintf('No data folder found at %s. No scripts executed.\n', dataFolder);
    return;
end

%% Execute generators
entries = dir(dataFolder);
folders = entries([entries.isdir] & ~startsWith({entries.name}, '.'));

for idx = 1:numel(folders)
    subfolderPath = fullfile(dataFolder, folders(idx).name);
    scripts = dir(fullfile(subfolderPath, 'generate_*.m'));
    for jdx = 1:numel(scripts)
        scriptPath = fullfile(subfolderPath, scripts(jdx).name);
        fprintf('Running %s\n', scriptPath);
        run(scriptPath);
    end
end

fprintf('All data generation scripts executed.\n');
end
