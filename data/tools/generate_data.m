% Runs all data generation scripts in the data folder and its subfolders

data_folder = fullfile(pwd, 'data');
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
