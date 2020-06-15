function simplePoseEstimation
%#codegen
%
% Copyright 2019 The MathWorks, Inc.

persistent simplePoseNet;
if isempty(simplePoseNet)
    simplePoseNet = coder.loadDeepLearningNetwork('simplePoseNet.mat','simplePoseNet');
end

hwobj = jetson; % To redirect to the code generatable functions.
w = webcam(hwobj,1,'640x480');
d = imageDisplay(hwobj);

threshold = 0.2;
downScaleFactor = 4;

while true
    % Capture the image from the webcam on hardware.
    Iin = snapshot(w);
    
    % Resize and crop the image to fit to the input of the network.
    Iinresize = imresize(Iin,[256 nan]);
    Itmp = Iinresize(:,(size(Iinresize,2)-192)/2:(size(Iinresize,2)-192)/2+192-1,:);
    Icrop = Itmp(1:256,1:192,1:3);
    
    % Detect keypoints
    output = simplePoseNet.predict(Icrop);
    [scores,idx_out] = max(output,[],[1,2],'linear');
    [y,x,~,~] = ind2sub(size(output),idx_out);
    keypoints = permute(cat(1,x,y),[3,1,2]);

    % Display image.
    Ioutcrop = visualizeKeyPoints(Icrop,keypoints*downScaleFactor,scores,threshold);
    Iout = imresize(Ioutcrop,2);
    image(d,Iout);
end

    function Iout = visualizeKeyPoints(I,joint,scores,threshold)
        SkeletonConnectionMap = [
            [16,14];[14,12];[17,15];[15,13];[12,13];[6,12];[7,13];[6,7];
            [6,8];[7,9];[8,10];[9,11];[2,3];[1,2];[1,3];[2,4];[3,5];[4,6];[5,7]
            ];
        
        persistent cmaps
        if isempty(cmaps)
            st = coder.load('cmaps.mat','cmaps');
            cmaps = st.cmaps;
        end
        scores = squeeze(scores);
        idn = find(scores > threshold);
        z = zeros(19,1);
        z(1:numel(idn)) = idn;
        if numel(idn) > 2
            idx1 = any(repmat(SkeletonConnectionMap(:,1),[1 19]) == repmat(z',[19,1]), 2);
            idx2 = any(repmat(SkeletonConnectionMap(:,2),[1 19]) == repmat(z',[19,1]), 2);
            idx  = idx1 & idx2;
            % Plot edges
            Pos1 = [joint(SkeletonConnectionMap(idx,1),1:2) joint(SkeletonConnectionMap(idx,2),1:2)];
            Iout = insertShape(I,"Line",Pos1,"LineWidth",3,"Color",cmaps(idx,:));
        else
            Iout = I;
        end
    end
end