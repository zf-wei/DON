% run_Burgers.m
% 主脚本依次运行 gen_Burgers_2.m, gen_Burgers_3.m, gen_Burgers_4.m

% 确保所有文件在 MATLAB 路径中
disp('Running gen_Burgers_2.m...');
gen_Burgers_2; % 运行 gen_Burgers_2.m

disp('Running gen_Burgers_3.m...');
gen_Burgers_3; % 运行 gen_Burgers_3.m

disp('Running gen_Burgers_4.m...');
gen_Burgers_4; % 运行 gen_Burgers_4.m

disp('All scripts executed successfully!');
