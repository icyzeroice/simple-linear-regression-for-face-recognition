				% config arguments
config.DirName = 'ORL_32x32';
config.ImageWidth = 32;
config.ImageHeight = 32;
NSAccuracyAll = [];
for iFileOrder = 1:10
  config.TestFileName = ['StTestFile', num2str(iFileOrder), '.txt'];
  config.TrainFileName = ['StTrainFile', num2str(iFileOrder), '.txt'];

				% computed arguments
  imageSize = config.ImageWidth .* config.ImageHeight;
  testMatrix = dlmread(strcat(config.DirName, '/',  config.TestFileName));
  trainMatrix = dlmread(strcat(config.DirName, '/',  config.TrainFileName));
  [nPicture, gndPosition] = size(testMatrix);
  [trainPicture, gndPosition] = size(trainMatrix);
  gnd = trainMatrix(:, gndPosition);
  nPerson = max(gnd);
				% arguments remained to be computed
				% iPicture shows the picture order
  iPicture = 1;
	       % Xi{iPerson}.m shows the picture m  belongs to iPerson
	       % Xi{iPerson}.n shows the amount of training
  Xi = cell(1,1);
  Xi{1}.mat = [];
  Xi{1}.num = 0;
				% TRAINING >>>>>
  while (iPicture <= trainPicture)
			   % iPerson shows whom the picture belongs to
    tempxi = trainMatrix(iPicture, 1:imageSize)';
    iPerson = trainMatrix(iPicture, gndPosition);
    				% init a matrix for everyone
    [rowNum, colNum] = size(Xi);
    if iPerson > colNum || isempty(Xi{iPerson}.mat)
      
      Xi{iPerson}.mat = tempxi;
      Xi{iPerson}.num = 0;
      tempAij = Xi{iPerson}.mat' * Xi{iPerson}.mat;
      pinvAij{iPerson} = pinv(tempAij);
      
    else

      a{iPerson} = Xi{iPerson}.mat' * tempxi;
      c = tempxi' * tempxi;
      t = 1 ./ (c - a{iPerson}' * pinvAij{iPerson} * a{iPerson});
      gama = pinvAij{iPerson} * a{iPerson};
      
				% ⎡pinvAij  0⎤
				% ⎢          ⎥
				% ⎣0        0⎦
      [rowPinvAij, colPinvAij] = size(pinvAij{iPerson});
      extendPinvAij = zeros(rowPinvAij+1, colPinvAij+1);
      extendPinvAij(1:rowPinvAij, 1:colPinvAij) = pinvAij{iPerson};
      size(extendPinvAij);
      
				%  ⎡gama⎤     T
				% t⎢    ⎥[gama  -1]
				%  ⎣-1  ⎦
      [rowGama, colGama] = size(gama);
      extendGama = -ones(rowGama + 1, colGama);
      extendGama(1:rowGama, 1:colGama) = gama;
      attachPinvAij = t .* extendGama * extendGama';

      pinvAij{iPerson} = extendPinvAij + attachPinvAij;
      Xi{iPerson}.mat = [Xi{iPerson}.mat, tempxi];
    end
    
    Xi{iPerson}.num = Xi{iPerson}.num + 1;

	  % if iPicture is the last picture, we jump out from the loop
    iPicture = iPicture + 1;
  end
				% initialize for testing data
  iPicture = 1;
  NSD = [];
  FSD = [];
  NFSD = [];
  D = {};
  NSAccuracy = 0;
  FSAccuracy = 0;
  NFSAccuracy = 0;
  while (iPicture <= nPicture)
    y = testMatrix(iPicture, 1:imageSize)';
    D{iPicture}.answer = testMatrix(iPicture, gndPosition);
				% compute the predictable y
    for iPerson = 1:nPerson
				% NS
      beta = pinvAij{iPerson} * Xi{iPerson}.mat' * y;
      predictY = Xi{iPerson}.mat * beta;
      NSd(iPerson) = norm(y - predictY);
    end
    [temp, D{iPicture}.NSRecognition] = min(NSd);
				% compute the accuracy
    if D{iPicture}.NSRecognition == D{iPicture}.answer
      NSAccuracy = NSAccuracy + 1;
    end
    iPicture = iPicture + 1;
  end

  NSAccuracyAll = [NSAccuracyAll, NSAccuracy ./ nPicture];
end
outputResult = NSAccuracyAll';
dlmwrite(['result_of_', config.DirName, '.txt'], outputResult);
