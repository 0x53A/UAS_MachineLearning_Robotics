clc; clear;

%number of epoch
numberOfEpoch = 10^6;
numberOfEpoch = 100000;
%number of data samples (vector of numberOfSamples rows and 1 column)
numberOfSamples = 100;
%our desired learning velocity
learningRate = 10^-3;

%create our input values randomly  
q1 = zscore(myRandVector(numberOfSamples,-10,10))';
q2 = zscore(myRandVector(numberOfSamples,-10,10))';
%create the grountruth data (measured data with the targeted/searched dynamics)
x1LabelVector = (q1 + q2 - 1)';
x2LabelVector = (q1 - q2 + 1 )';
%create our vector of labels (measured data)
xLabelVector = [x1LabelVector x2LabelVector];

%initialization of our vector of weights
w = myRandVector(18,-1,1);
%our vector of input data (with numberOfSamples rows and two columnn).
uVector = [q1' q2'];
%uVector = zscore(uVector);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%NEW%%%%%%%%%%%%%%%%%%%%
alpha = 0.001; % = r
mBeta_1 = 0.9;    % decay rate w.r.t first moment
mBeta_2 = 0.999;  % decay rate w.r.t second moment 
epsilon = 1e-8; % against zero division
mk = zeros(size(w)); % first moment estimate
v_k = zeros(size(w)); % second moment estimate
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% n = size(uVector,1);
% for i=1:n
% uVector(i,:) = zscore(uVector(i,:));
% end
% we go through the entire dataset numberOfEpoch times in the following loop
for i = 1:numberOfEpoch
    %The three key steps of the training objective

    %Step (1) 
    %Obtain the output of the network for all samples and the current
    %vector of weights being optimized
    [x1Vector, x2Vector] = forwardVector (uVector, w);
    
    %Step (2) 
    % Compute the gradient vector associated with the
    % current input (the samples) of the network and associated 
    % labels (measured data) as well as the current weight
    % vector (the w vector). Observe that the returned Loss is w.r.t all
    % the samples
    [grad, returnedLoss] = calcGradientOfLossFunction(uVector, xLabelVector, w);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%NEW%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % w.r.t first
    mk = mBeta_1 * mk + (1 - mBeta_1)*grad;

    % w.r.t second
    v_k = mBeta_2 * v_k + (1 - mBeta_2)*(grad.^2);

    % corrective adaptation
    mHat_k = mk/(1 - mBeta_1^i);
    vHat_k = v_k/(1 - mBeta_2^i);

    % adaptive learning rate
    rk = alpha./ (sqrt(vHat_k) + epsilon);

    % weight update
    w = w - rk.* mHat_k; %instead of wk = wk - r*grad;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    %Step (3) 
    % Optimize the weights of the neural network using the previously calculated gradient vector.
    % Once this has been done, jump to Step (1) and again and gain to steer
    % the weights (w) toward a direction for which the neural network is going
    % to capture the underlying model entailed in the data.
   % w = w - learningRate*grad; %%%%%%%%%%%%NEW this line must be
   % deactivated (sine we are optimizing the parameters in line 79)
    
    % The following is just for visulaization purposes, making sure that the
    % user is aware of the current state of the parameter (i.e., weight) optimization
    % process
    lossValue = loss (x1LabelVector, x2LabelVector, x1Vector, x2Vector);
    fprintf("step: %i ===== loss:  %10.10f \n",i, returnedLoss);
end

% This function is going to compute the loss by using input, labels and
% frozen weights (that is, it does not modify the w)
function [lossResult] = lossFromForwardVector (uVector, xLabels, w)
    [x1Vector, x2Vector] = forwardVector (uVector, w);
    lossResult = loss (xLabels(:,1), xLabels(:,2), x1Vector, x2Vector);
end

%This function computes the output of our neural network for given inputs (i.e., vector of single inputs),
%and vector of weights (i.e., w) and returns the vector of corresponding outputs
function [x1Vector, x2Vector] = forwardVector (uVector, w)
    n = size(uVector, 1);
    x1Vector = zeros(n,1);
    x2Vector = zeros(n,1);
    for i = 1:n
         [x1,x2] = forward(uVector(i,:), w);
         x1Vector(i) = x1;
         x2Vector(i) = x2;
    end
end

%This function computes the output of our neural network for given inputs (i.e., single input and output values),
%and vector of weights (i.e., w) and returns the vector of corresponding single
%outputs (i.e., two sclars)
function [x1, x2] = forward (u, w)
q1 = u(1);
q2 = u(2);

a = 0.7;
b = 0.3;
%Hidden layer 1;
w11_0 = w(1); w12_0 = w(2); w1b_0 = w(3); w21_0 = w(4); w22_0 = w(5); w2b_0 = w(6); 
%hidden layer 2;
w11_1 = w(7); w12_1 = w(8); w1b_1 = w(9); w21_1 = w(10); w22_1 = w(11); w2b_1 = w(12); 
%output layer
w11_2 = w(13); w12_2 = w(14); w1b_2 = w(15); w21_2 = w(16); w22_2 = w(17); w2b_2 = w(18); 

x1_1 = myActivation(w11_0*q1 + w21_0*q2 + w1b_0, a,b);
x2_1 = myActivation(q2*w22_0 + q1*w12_0 + w2b_0, a,b);

x1_2 = myActivation(w11_1*x1_1 + w21_1*x2_1 + w1b_1, a,b);
x2_2 = myActivation(x2_1*w22_1 + x1_1*w12_1 + w2b_1, a,b);

x1 = myActivation(w11_2*x1_2 + w21_2*x2_2 + w1b_2, a,b);
x2 = myActivation(x2_2*w22_2 + x1_2*w12_2 + w2b_2, a,b);

end

%The activation function, you might want to extend this function to take
%additional activitation functions under account
function [outputParameter]=myActivation(inputParameter, a, b)

    if(inputParameter > 0)
        outputParameter = a*inputParameter;
    else
        outputParameter = b*inputParameter; 
    end
end

%This function compute the loss and the sum of the single losses over all
%the data samples
function [rloss] = loss (x1LabelVector, x2LabelVector, X1Vector, X2Vector)
%x1LabelVector = grounturth X1
%x2LabelVector = grounturth X2
%X1Vector = x1 output of the network
%X2Vector = x2 output of the network

rloss = sum( (x1LabelVector - X1Vector).^2 + (x2LabelVector - X2Vector).^2 );

end

%This function compute the gradient of the loss w.r.t to the weights (i.e., the w)
function [gradient, loss] = calcGradientOfLossFunction(inputTrainingVector, outputLabelVector, w)
    % Calculate the gradient of the loss function
    % with respect to the parameters (trained Weights)
    
    % Define a small step size 
    espsilonValue = 1e-6;
    
    trainedWeights = w;

    % Initialize the gradient vector
    gradient = zeros(size(trainedWeights)); 
    
    % Compute the contribution to the gradient of the loss for each weight
    for i = 1:length(trainedWeights)
        % Perturb the current weight using the espsilonValue to the right
        trainedWeightsPerturbedRight = trainedWeights;
        trainedWeightsPerturbedRight(i) = trainedWeightsPerturbedRight(i) + espsilonValue;
        
        % Calculate loss values for wright-perturbed parameter
        [costFunctionPertubedRight]= lossFromForwardVector(inputTrainingVector,outputLabelVector, trainedWeightsPerturbedRight);
        
        % Perturb the parameter value by -espsilonValue (ie, to the left)
        trainedWeightsPerturbedLeft = trainedWeights;
        trainedWeightsPerturbedLeft(i) = trainedWeightsPerturbedLeft(i) - espsilonValue;
        
        % Calculate loss values for perturbed parameter
        [costFunctionPertubedLeft] = lossFromForwardVector(inputTrainingVector, outputLabelVector, trainedWeightsPerturbedLeft);
        
        % Approximate the gradient contribution 
        gradient(i) = (costFunctionPertubedRight - costFunctionPertubedLeft)... 
        / (2 * espsilonValue);
        %pick up the loss
        if(i == length(trainedWeights))
            loss = costFunctionPertubedLeft;
        end
    end
end

%quick test
q1 = 0.75;
q2 = -0.25;
% u = zscore([q1 q2]);
% q1 = u(1);
% q2 = u(2);
%we expect x1 = q1 + q2 -1 = -0.5 and x2 = q1 - q2 +1 = 2

[x1, x2]=forward ([q1,q2], w);
fprintf("Quick Verification: x1= %10.10f ===== x2=  %10.10f \n",x1, x2);
