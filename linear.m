averagePercentNearest = [];
averagePercentFarthest = [];
averagePercentBoth = [];

for fileOrder=1:1
				% direction config
  dir = 'ORL_64x64/';
  picSize = 64;
  testFileName = 'StTestFile';
  trainFileName = 'StTrainFile';
  format = '.txt';
  oneLength = picSize.^2;
  oneLengthGnd = oneLength+1;
				% read matrix files
  getFileOrder = num2str(fileOrder);
  testFile = strcat(dir, testFileName, getFileOrder, format);
  trainFile = strcat(dir, trainFileName, getFileOrder, format);
				% testing data
  TestMatrix = dlmread(testFile);
  getTestSize = size(TestMatrix);
  testDataLength = getTestSize(1);
				% training data
  TrainMatrix = dlmread(trainFile);
  getTrainSize = size(TrainMatrix);
  trainDataLength = getTrainSize(1);
  trainKinds = TrainMatrix(trainDataLength, oneLengthGnd);
  trainEveryone = trainDataLength./trainKinds;
				% initial
  tempXi = zeros(trainEveryone, oneLength);
  Xi = [];
  X = [];
  Bi = [];
				% the total number of persons
  for personNum = 1:trainKinds
				% the training time for everyone
    for trainingTime = 1:trainEveryone
      tempXi(trainingTime,:) = TrainMatrix(((personNum-1)*trainEveryone+trainingTime), 1:oneLength);
    end
				% pretreatment
    % tempMax = max(tempXi(:));
    % Xi(personNum).m = tempXi'./tempMax;
    Xi(personNum).m = tempXi';
    Xi(personNum).personNum = personNum;
    X = [X, Xi(personNum).m];
  end

  for personNum = 1:trainKinds
    if personNum-1
      Bipart1 = X(:, 1:trainEveryone*(personNum-1));
    else
      Bipart1 = [];
    end
    if trainKinds-personNum
      Bipart2 = X(:, (trainEveryone*personNum+1):trainEveryone*trainKinds);
    else
      Bipart2 = [];
    end
    Bi(personNum).m = [Bipart1, Bipart2];
  end
  
  
  trueTimeNearest = 0;			% linear predict
  trueTimeFarthest = 0;
  trueTimeBoth = 0;
  for testTime = 1:testDataLength
    y = TestMatrix(testTime, 1:oneLength)';
				% pretreatment
    % tempMax = max(y(:));
    % y = y./tempMax;
    for personNum = 1:trainKinds
      cacheXi = Xi(personNum).m;
      cacheBi = Bi(personNum).m;
      
      cacheXiT = cacheXi';
      tempPinv = pinv(cacheXiT*cacheXi);
      yPredict = cacheXi*tempPinv*cacheXiT*y;
      
      cacheBiT = cacheBi';
      tempPinv = pinv(cacheBiT*cacheBi);
      yFarPredict = cacheBi*tempPinv*cacheBiT*y;
      
				% >>> nearest distence
      tempD = norm(yPredict-y);
      				% >>> farthest distence
      tempFarD = norm(yFarPredict-y);
				% get person order
      if personNum==1
        minD = tempD;
        maxD = tempFarD;
        judgeD = tempD./tempFarD;
        pN = 1;
        pNFar = 1;
        pNBoth = 1;
     else
      if tempD < minD
        minD = tempD;
        pN = personNum;
      end
      
      if tempFarD > maxD
        maxD = tempFarD;
        pNFar = personNum;
      end

      if tempD./tempFarD < judgeD
        judgeD = tempD./tempFarD;
        pNBoth = personNum;
      end
     end
    end
    
    testnum = TestMatrix(testTime, oneLengthGnd);
    if pN==TestMatrix(testTime, oneLengthGnd)
      trueTimeNearest+=1;
    end
    if pNFar==TestMatrix(testTime, oneLengthGnd)
      trueTimeFarthest+=1;
    end
    if pNBoth==TestMatrix(testTime, oneLengthGnd)
      trueTimeBoth+=1;
    end
  end
		  % the accuracy everytime -------------------> output
  fprintf('%s%c%s\r\n', '<-VVV---------<', getFileOrder, '>---------VVV->');
  truePercentNearest = trueTimeNearest./testDataLength
  averagePercentNearest = [averagePercentNearest, truePercentNearest];
  truePercentFarthest = trueTimeFarthest./testDataLength
  averagePercentFarthest = [averagePercentFarthest, truePercentFarthest];
  truePercentBoth = trueTimeBoth./testDataLength
  averagePercentBoth = [averagePercentBoth, truePercentBoth];
  fprintf('%s%c%s\r\n\r\n', '<-^^^---------<', getFileOrder, '>---------^^^->');
      
end
				% the average accuracy ---------------------> output
averagePercentNearest = mean(averagePercentNearest(:))
averagePercentFarthest = mean(averagePercentFarthest(:))
averagePercentBoth = mean(averagePercentBoth(:))
