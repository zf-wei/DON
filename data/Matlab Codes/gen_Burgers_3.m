% number of realizations to generate
N = 1100;

% parameters for the Gaussian random field
gamma = 4;
tau = 5;
sigma = 25;

% viscosity
visc = 0.001;

% grid size
s = 4096; % 空间估值点 比较精细 最后输出是粗糙的
steps = 100; % 向前计算100步
nn = 101; % 空间估值点 粗糙输出

input = zeros(N, nn);
if steps == 1
    output = zeros(N, s); % 如果只输出一步 那就输出精细的
else
    output = zeros(N, steps, nn); % 否则输出粗糙的以节约存储
end

tspan = linspace(0,1,steps+1);
x = linspace(0,1, s+1);
X = linspace(0,1, nn);

for j=1:N
    u0 = GRF(s/2, 0, gamma, tau, sigma, "periodic");
    u = Burgers(u0, tspan, s, visc); % 使用谱方法解方程，初始值也会被重新评估
    
    u0_eval = u0(X);
    input(j,:) = u0_eval;
    
    if steps == 1
        output(j,:) = u.values;
    else
        for k=1:(steps+1) % 因为初始值会被重新计算，所以这儿是 101
            output(j,k,:) = u{k}(X);
        end
    end
    
    disp(j);
end
save('Burgers_0.001.mat', 'input', 'output', 'tspan',  'gamma', 'tau', 'sigma')

