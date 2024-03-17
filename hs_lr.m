%loading the data and creatring seperate matrices for inputs and outputs
load CWnormalised.csv 
hsIn = CWnormalised(:,1:4)';
hsOut = CWnormalised(:,5)';  

% Create a parallelpool
 pool = parpool;

%creating plots to identify outliers
cols = 4;
figure;
for i = 1:cols-1
    for j = i+1:cols
        subplot(cols-1, cols-1, (i-1)*(cols-1) + j-1);
        plot(hsIn(i, :), hsIn(j, :), 'ro');
        title([num2str(i), 'vs', num2str(j)]);
    end
end
%............................................ exploration of different parameters ..............................................................
%Topological Exploration - testing out combinations of different number of hidden layers and hidden units 3 times each
net.trainFcn = 'traingd';  
perfMat_topology = [];

parfor i = 1:20 % Use parfor for parallel processing
    hl_1_units = i * 1; 
    top_comb_perf = [];
    for j = 1:5
        net = fitnet([hl_1_units]);
        net = init(net);
        net.trainParam.epochs = 1000;
        net = train(net, hsIn, hsOut);
        t = net(hsIn);
        top_comb_perf = [top_comb_perf, mse(net, t, hsOut)];
    end
    perfMat_topology = [perfMat_topology; top_comb_perf];
end
% Close the parallel pool
delete(pool);
% Plot each column against the row index
figure;
for col = 1:size(perfMat_topology, 2)
    subplot(size(perfMat_topology, 2), 1, col);
    plot(1:size(perfMat_topology, 1), perfMat_topology(:, col));
    xlabel('Row Index');
    ylabel(['Values in Column ', num2str(col)]);
    title(['Plot of Column ', num2str(col)]);
end
% Adjust subplot spacing
sgtitle('Plots of Values in Each Column Against Row Index');
%...............................................exploration of different data splicing ratios.......................................................................................................
perfMat_act_fcn = [];
activation_functions = {'tansig', 'logsig', 'elliotsig', 'poslin', 'purelin', 'radbas', 'satlins', 'tribas'};
parfor i = 1:length(activation_functions)
    current_act_fcn = activation_functions{i};
    act_fcn_perf = [];
    for j = 1:5
        net = fitnet([9]);
        net.layers{1}.transferFcn = current_act_fcn;
        net.layers{2}.transferFcn = 'purelin';  %constant output layer
        net = init(net);
        net.trainParam.epochs = 1000;
        net = train(net, hsIn, hsOut);
        t = net(hsIn);
        act_fcn_perf = [act_fcn_perf, mse(net, t, hsOut)];
    end
    perfMat_act_fcn = [perfMat_act_fcn; act_fcn_perf];
end
perfMat_act_fcn;
% Close the parallel pool
delete(gcp);

% Calculate the mean of each column
matrix = perfMat_act_fcn'
meanValues = mean(matrix, 1);
disp('Mean of each column:');
disp(meanValues);
disp(activation_functions);
%...............................................exploration of different activation(transfer) functions.........................................................................................
perfMat = [];

%first ratio split - 80, 10, 10
net = fitnet([9]);
net = init(net); 
net.layers{1}.transferFcn = 'radbas';   %hidden layer 1
net.layers{2}.transferFcn = 'purelin';  %output layer

net.trainFcn = 'traincgp';       %training function 
net.performFcn = 'mse';         %mean-squared error function
net.divideFcn = 'dividerand';   %how to divide data into training, validation and testing
net.divideParam.trainRatio = 80/100;    %training set
net.divideParam.valRatio = 10/100;      %validation set
net.divideParam.testRatio = 10/100;     %testing set

[net,tr] = train(net, hsIn, hsOut); % Trains the NN net using the input data hsIn and output data hsOut. 
                                      % The training process returns the now trained network in 'net' and training record information in 'tr'.

simT = sim(net ,hsIn); %sim function applies the newly trained NN 'net' to the inputs 'hsIn' and stores the outputs in the variable 'simT'. The outputs represents the NN's predictions on the input data.
perf = perform(net,hsIn,hsOut); %perform function calculates a perfromance metric (mse)
perfMat = [perfMat, perf] %appends perf to the end of the perfMat array
%cross validation - k-fold
xy = [CWnormalised(:,1) CWnormalised(:,2) CWnormalised(:,3) CWnormalised(:,4)];
cvtree = fitrtree(xy,hsOut,'CrossVal','on');
mse_loss = kfoldLoss(cvtree)

%second ratio split - 70, 15, 15

net = fitnet([9]);
net = init(net); 
net.layers{1}.transferFcn = 'radbas';   %hidden layer 1
net.layers{2}.transferFcn = 'purelin';  %output layer

net.trainFcn = 'traincgp';       %training function 
net.performFcn = 'mse';         %mean-squared error function
net.divideFcn = 'dividerand';   %how to divide data into training, validation and testing
net.divideParam.trainRatio = 70/100;    %training set
net.divideParam.valRatio = 15/100;      %validation set
net.divideParam.testRatio = 15/100;     %testing set

[net,tr] = train(net, hsIn, hsOut); % Trains the NN net using the input data hsIn and output data hsOut. 
                                      % The training process returns the now trained network in 'net' and training record information in 'tr'.

simT = sim(net ,hsIn); %sim function applies the newly trained NN 'net' to the inputs 'hsIn' and stores the outputs in the variable 'simT'. The outputs represents the NN's predictions on the input data.
perf = perform(net,hsIn,hsOut); %perform function calculates a perfromance metric (mse)
perfMat = [perfMat, perf] %appends perf to the end of the perfMat array
%cross validation - k-fold
xy = [CWnormalised(:,1) CWnormalised(:,2) CWnormalised(:,3) CWnormalised(:,4)];
cvtree = fitrtree(xy,hsOut,'CrossVal','on');
mse_loss = kfoldLoss(cvtree)

%third ratio split - 60, 20, 20
net = fitnet([9]);
net = init(net); 
net.layers{1}.transferFcn = 'radbas';   %hidden layer 1
net.layers{2}.transferFcn = 'purelin';  %output layer

net.trainFcn = 'traincgp';       %training function 
net.performFcn = 'mse';         %mean-squared error function
net.divideFcn = 'dividerand';   %how to divide data into training, validation and testing
net.divideParam.trainRatio = 60/100;    %training set
net.divideParam.valRatio = 20/100;      %validation set
net.divideParam.testRatio = 20/100;     %testing set

[net,tr] = train(net, hsIn, hsOut); % Trains the NN net using the input data hsIn and output data hsOut. 
                                      % The training process returns the now trained network in 'net' and training record information in 'tr'.

simT = sim(net ,hsIn); %sim function applies the newly trained NN 'net' to the inputs 'hsIn' and stores the outputs in the variable 'simT'. The outputs represents the NN's predictions on the input data.
perf = perform(net,hsIn,hsOut); %perform function calculates a perfromance metric (mse)
perfMat = [perfMat, perf] %appends perf to the end of the perfMat array
%cross validation - k-fold
xy = [CWnormalised(:,1) CWnormalised(:,2) CWnormalised(:,3) CWnormalised(:,4)];
cvtree = fitrtree(xy,hsOut,'CrossVal','on');
mse_loss = kfoldLoss(cvtree)
%...............................................Final Optimised Neural Network......................................................................................................
perfMat = []
net = fitnet([9]);
net = init(net); 
net.layers{1}.transferFcn = 'radbas';   %hidden layer 1
net.layers{2}.transferFcn = 'purelin';  %output layer

net.trainFcn = 'traincgp';       %training function 
net.performFcn = 'mse';         %mean-squared error function
net.divideFcn = 'dividerand';   %how to divide data into training, validation and testing
net.divideParam.trainRatio = 80/100;    %training set
net.divideParam.valRatio = 10/100;      %validation set
net.divideParam.testRatio = 10/100;     %testing set

[net,tr] = train(net, hsIn, hsOut); % Trains the NN net using the input data hsIn and output data hsOut. 
                                      % The training process returns the now trained network in 'net' and training record information in 'tr'.
disp('optimised model results')
simT = sim(net ,hsIn); %sim function applies the newly trained NN 'net' to the inputs 'hsIn' and stores the outputs in the variable 'simT'. The outputs represents the NN's predictions on the input data.
perf = perform(net,hsIn,hsOut); %perform function calculates a perfromance metric (mse)
perfMat = [perfMat, perf] %appends perf to the end of the perfMat array
%cross validation - k-fold
xy = [CWnormalised(:,1) CWnormalised(:,2) CWnormalised(:,3) CWnormalised(:,4)];
cvtree = fitrtree(xy,hsOut,'CrossVal','on');
mse_loss = kfoldLoss(cvtree)