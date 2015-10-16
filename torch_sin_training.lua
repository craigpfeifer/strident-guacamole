require "nn"

-- generate randomized training data for sin function

dataset={};
function dataset:size() return 200 end
for i=1,dataset:size() do

  local first_rand = math.random(10)
  local second_rand = math.random(10)

  local input = torch.Tensor(1);
  input[1] = second_rand * (math.pi / first_rand)

  local output = torch.Tensor(1);
  output[1] = torch.sin(input[1]);

  print (second_rand.." * (pi / "..first_rand.." ) "..input[1].." : "..output[1])
  dataset[i] = {input, output}
end

-- setup NN with single hidden layer
-- tanh() transfer function
-- mean squared error metric
-- SGD training function

hid_units = 70;
mlp = nn.Sequential()
mlp:add( nn.Linear(1, hid_units) )
mlp:add( nn.Tanh() ) -- hyperbolic tangent transfer function
mlp:add( nn.Linear(hid_units, 1) )

criterion = nn.MSECriterion() -- Absolute Error criterion
trainer = nn.StochasticGradient(mlp, criterion)
trainer.maxIteration = 100
trainer:train(dataset)

test = torch.Tensor(1)

test_vals={};
test_vals[0] = 0;               -- 0
test_vals[1] = math.pi/4;       -- pi / 4
test_vals[2] = math.pi/2;       -- pi / 2
test_vals[3] = 3*(math.pi/4);   -- 3 pi / 4
test_vals[4] = math.pi;         -- pi
test_vals[5] = 5*(math.pi/4);   -- 5 pi / 4
test_vals[6] = 3*(math.pi/2);   -- 3 pi / 2
test_vals[7] = 7*(math.pi/4);   -- 7 pi / 4
test_vals[8] = 2*(math.pi);     -- 2 pi

for i,test_val in pairs(test_vals) do

  test[1] = test_val;
  expected = torch.sin(test[1]);

  print ("TEST IN  : "..test[1]);
  print ("EXPECTED : "..expected);
  y = mlp:forward(test);
  print ("COMPUTED : "..y[1])
  err = y - expected;
  sqerr = math.pow(err[1],2);
  print ("   error : "..err[1]);
  print ("sq error : "..sqerr.."\n");

end
