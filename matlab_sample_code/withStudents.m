

q1 = myRandVector(1000,-1,1)';
q2 = myRandVector(1000,-1,1)';

u = [q1; q2];
w = myRandVector(18,-1,1);

epoch = 10^6;

for i = 1:epoch
    for j=1:1000
     [x1, x2]=forward(u(:,j)*0,w);
     disp([x1, x2]);
    end
end

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

function [outputParameter]=myActivation(inputParameter, a, b)
    if(inputParameter > 0)
        outputParameter = a*inputParameter;
    else
        outputParameter = b*inputParameter; 
    end
end