classdef PoseEstimator
    % Pose Estimator class

    % Copyright 2020 The MathWorks, Inc.
    
    properties
        Threshold = 0.2 % Confidence threshold at detection
    end
    
    properties (SetAccess=private)
        Network % An object providing an inteface to the Network
        SkeletonConnectionMap
        InputSize = [256,192,3] % Input image size
        OutputSize = [64,48] % Output heatmap size
        NumKeypoints = 17 % Number of keypoints
        DownScaleFactor = 4 % Down scale factor
    end
    
    methods
        function obj = PoseEstimator(options)
            % Create an posenet.PoseEstimator
            arguments
                options.MATFile (1,1) string = "simplePoseNet.mat"
                options.NetworkName (1,1) string = "simplePoseNet"
                options.SkeletonConnectionMap (1,1) string = "SkeletonConnectionMap"
            end
            
            % Load the network
            d = load(options.MATFile);
            obj.Network = getfield(d,options.NetworkName); %#ok<GFLD>
            obj.SkeletonConnectionMap = getfield(d,options.SkeletonConnectionMap); %#ok<GFLD>
            % Identify the input layer (1 at most case)
            idxInputLayer = find(strcmp({obj.Network.Layers.Name},obj.Network.InputNames{1}));
            obj.InputSize = obj.Network.Layers(idxInputLayer(1)).InputSize;
            output = predict(obj.Network,zeros(obj.InputSize));
            obj.NumKeypoints = size(output,3);
            obj.OutputSize = size(output,1:2);
            obj.DownScaleFactor = obj.InputSize(1) ./ obj.OutputSize(1);
        end
        
        function [croppedImages, croppedBBoxes] = normalizeBBoxes(obj,images, bboxes)
            % Normalize the images to fit the network input keeping the
            % aspect ratio.
            targetSize = [obj.InputSize(2), obj.InputSize(1)];
            croppedImages = zeros(targetSize(2),targetSize(1),3,size(bboxes,1),...
                class(images));
            croppedBBoxes = bboxes;
            for k = 1:size(bboxes,1)
                bbox = bboxes(k,:);
                heightMajor = bbox(4) > bbox(3);
                maxEdge = max(bbox(4),bbox(3));
                scale = targetSize(heightMajor+1)/maxEdge*0.9;
                normalizedTransform = eye(3);
                normalizedTransform(1,1) = scale;
                normalizedTransform(2,2) = scale;
                croppedBBoxes(k,:) = [bbox(1)-(targetSize(1)/scale-bbox(3))/2,...
                    bbox(2)-(targetSize(2)/scale-bbox(4))/2,...
                    targetSize(1)/scale,...
                    targetSize(2)/scale];
                moveToTopLeftTr = eye(3);
                moveToTopLeftTr(3,1) = -croppedBBoxes(k,1);
                moveToTopLeftTr(3,2) = -croppedBBoxes(k,2);
                Tout = moveToTopLeftTr * normalizedTransform;
                tform = affine2d(Tout);
                croppedImages(:,:,:,k) = imwarp(images,tform,...
                    'OutputView',imref2d([targetSize(2) targetSize(1)]),...
                    'SmoothEdges',true,...
                    'FillValues',0);
            end
        end
        
        function keypoints = detectPose(obj,images)
            heatmaps = predict(obj,images);
            keypoints = heatmaps2Keypoints(obj,heatmaps);
        end
        
        function heatmaps = predict(obj,croppedImages)
            % Predict heatmaps.
            heatmaps = predict(obj.Network,croppedImages);
        end
        
        function keypoints = heatmaps2Keypoints(obj,heatmaps)
            % Find the maximum points in each heatmap.
            % TODO: implement sub-pixel estimation with neibor pixels using bilinear
            % interpolation.
            threshold = obj.Threshold;
            [confs,idx] = max(heatmaps,[],[1,2],'linear');
            [y,x,~,~] = ind2sub(size(heatmaps),idx);
            keypoints = permute(cat(1,x,y),[3,1,4,2]) * obj.DownScaleFactor;
            keypoints = [keypoints,permute(confs>threshold,[3 1 4 2])];
        end
        
        function Iout = visualizeHeatmaps(obj,heatmaps, I)
            arguments
                obj
                heatmaps
                I uint8
            end
            I = reshape(rgb2gray(reshape(permute(I,[1 2 4 3]),1,[],3)),size(I,1),size(I,2),1,[]);
            % Create color mask
            cmap = shiftdim(hsv(size(heatmaps,3))',-2);
            mask = imresize(max(heatmaps,[],3),obj.DownScaleFactor);
            mask = max(mask,0);
            mask = min(mask,1);
            outputColored = squeeze(max(permute(heatmaps,[1,2,5,3,4]).* cmap,[],4));
            % Overlay
            outputColoredUpscaled = imresize(outputColored,obj.DownScaleFactor);
            Iout = (1-mask).*im2double(I)*0.5 + mask.*outputColoredUpscaled;
            Iout = im2uint8(Iout);
        end
        
        function Iout = visualizeKeyPoints(obj,I,joints)
            skeleton = obj.SkeletonConnectionMap;
            numEdges = size(skeleton,1);
            cmapEdges = im2uint8(hsv(numEdges));
            
            % Plot edges and nodes
            Iout = I;
            
            for k = 1:size(joints,3)
                pts = joints(:,:,k);
                pos = [pts(skeleton(:,1),:) pts(skeleton(:,2),:)];
                validIdxEdge = all(pos(:,[3 6])>0, 2);
                
                pos = pos(validIdxEdge,[1,2,4,5]);
                cmaps_temp = cmapEdges(validIdxEdge,:);
                
                Iout(:,:,:,k) = insertShape(Iout(:,:,:,k),...
                    "Line",pos,"LineWidth",3,"Color",cmaps_temp);
                validIdxNode = pts(:,3) > 0;
                Iout(:,:,:,k) = insertShape(Iout(:,:,:,k),...
                    "FilledCircle",[pts(validIdxNode,1:2) ones(sum(validIdxNode),1)*3],...
                    "Color","Red");
            end
        end
        
        function Iout = visualizeKeyPointsMultiple(obj,I,joints,bboxes)
            arguments
                obj
                I (:,:,:,1) uint8
                joints
                bboxes
            end
            skeleton = obj.SkeletonConnectionMap;
            numEdges = size(skeleton,1);
            cmapEdges = im2uint8(hsv(numEdges));
            targetSize = [obj.InputSize(2), obj.InputSize(1)];
            
            % Move to the original positions in the original image
            % coordinate.
            joints = [joints(:,1:2,:).*permute(bboxes(:,3:4)./targetSize,[3,2,1])...
                + permute(bboxes(:,1:2),[3,2,1]),joints(:,3,:)];
            
            % Plot edges and nodes
            Iout = I;
            
            % TODO: vectorize this.
            for k = 1:size(joints,3)
                pts = joints(:,:,k);
                pos = [pts(skeleton(:,1),:) pts(skeleton(:,2),:)];
                validIdxEdge = all(pos(:,[3 6])>0,2);
                
                pos = pos(validIdxEdge,[1,2,4,5]);
                cmaps_temp = cmapEdges(validIdxEdge,:);
                
                Iout = insertShape(Iout,"Line",pos,"LineWidth",3,"Color",cmaps_temp);
                validIdxNode = pts(:,3)>0;
                Iout = insertShape(Iout,...
                    "FilledCircle",[pts(validIdxNode,1:2) ones(sum(validIdxNode),1)*3],...
                    "Color","Red");
            end
        end
        
    end
end