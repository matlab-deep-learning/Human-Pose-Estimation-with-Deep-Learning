function multiPersonPoseEstimation
%#codegen
%
% Copyright 2023 The MathWorks, Inc.

persistent simplePoseNet;
persistent detector;
if isempty(simplePoseNet)
    simplePoseNet = coder.loadDeepLearningNetwork('simplePoseNet.mat','simplePoseNet');
end
if isempty(detector)
    % Run the following commands if you face an error regarding lack of
    % 'object_detector.mat' file. Note that yolov4ObjectDetector requires
    % Computer Vision Toolbox™ Model for YOLO v4 Object Detection support package
    % >> detector =  yolov4ObjectDetector('tiny-yolov4-coco')
    % >>save('object_detector','detector');
    detector = coder.loadDeepLearningNetwork('object_detector.mat');
end

hwobj = jetson; % To redirect to the code generatable functions.
w = webcam(hwobj,1,'1280x720');
d = imageDisplay(hwobj);

threshold = 0.2;
downScaleFactor = 4;

while true
    % Capture the image from the webcam on hardware.
    Iin = snapshot(w);

    % Detect persons
    [bboxes, ~, labels] = detect(detector, Iin);

    % Select 'person' class
    index = labels == "person";
    bboxes = bboxes(index,:);

    % Inflate the bounding box for pose estimation
    scale = 1.1;
    topLeft = bboxes(:,1:2) + (1-scale)/2*bboxes(:,3:4);
    wh = scale*bboxes(:,3:4);
    bboxesInflated = [topLeft, wh];

    Iout = insertShape(Iin,"rectangle",bboxesInflated);
    for k = 1:size(bboxesInflated,1)
        % Warp the input image to be fed into the network using for each bounding box
        bbox = bboxesInflated(k,:);
        bbox(1:2) = bbox(1:2) - 1;
        if bbox(3) > bbox(4)
            scale = 192/bbox(3);
        else
            scale = 256/bbox(4);
        end
        offset = -bbox(1,1:2);
        tform = simtform2d(scale,0,offset*scale);
        cropLim = floor(bbox([2,1])+bbox([4,3]));
        cropLim(1) = max(1,min(size(Iin,1),cropLim(1)));
        cropLim(2) = max(1,min(size(Iin,2),cropLim(2)));
        Icrop = imwarp(Iin(1:cropLim(1),1:cropLim(2),:),tform,'OutputView',imref2d([256,192]));

        % Detect keypoints
        Icrop = Icrop(1:256,1:192,1:3);
        output = simplePoseNet.predict(Icrop);
        [scores,idx_out] = max(output,[],[1,2],'linear');
        [y,x,~,~] = ind2sub(size(output),idx_out);
        keypoints = permute(cat(1,x,y),[3,1,2]);
        keypoints = keypoints * downScaleFactor;

        % Revert the keypoint coordinates to the original coordinates
        keypoints = tform.transformPointsInverse(keypoints);
    
        % Visualize key points
        Iout = visualizeKeyPoints(Iout,keypoints,scores,threshold);
    end
    % Mirror
    Iout = Iout(:,end:-1:1,:);
    % Display image.
    image(d,permute(Iout,[2,1,3]));
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
            Iout = insertShape(I,"Line",Pos1,"LineWidth",5,"Color",cmaps(idx,:));
        else
            Iout = I;
        end
    end
end