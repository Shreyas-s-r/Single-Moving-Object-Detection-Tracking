%detect
clc ;
close all;
clear;
%% read the video file
vid = VideoReader('demo.avi');
% get the properties of the video file
nframes = get(vid,'NumFrames');
h = vid.Height;
w = vid.Width;
%% assign of background frame 
% intialize to zero across (h,w,dimention-3 i.e. (R,G,B) )
Imzero = zeros(h,w,3); 
% compute the background image
for i = 1:10
 % Integers can only be combined with integers of the same class, or scalar doubles.
 % so convert the read frames to double as 'Imzero is of type double'
 Img{i} = double(read(vid,i));
 Imzero = Img{i}+Imzero;
end
% taking average of first 10 frames and take it as reference or back-ground frame
Background = Imzero/10;
%% initialization of the terms used in Kalman filter 
% measurement error or sensor noise co-variance matrix
R=[[0.2845,0.0045]',[0.0045,0.0455]'];
% process noise co-variance matrix
Q=0.01*eye(4);
% time for 1 cycle 
dt=1;
% adaptation matrix A, B and H - to make dimension of matrix to perform operations
% like +,-* and /
A=[[1,0,0,0]',[0,1,0,0]',[dt,0,1,0]',[0,dt,0,1]'];
B = [1/2*dt*dt,0,dt,0;0,1/2*dt*dt,0,dt]';
H= [[1,0]',[0,1]',[0,0]',[0,0]'];
% assuming that no new velocity is added to u(control variable matrix)except initial velocity
u = [0;0];
% process co-variance matrix - is initialized 
P = 100*eye(4);
% to set the position of prediction to centre of the frame at the beginning state 
kinit=0;
% initialized state matrix to zero 
x=zeros(100,4);
%% loop over all images
for i=1:nframes
 
 % load image
 Img = (read(vid,i)); 
 imshow(Img)
 % convert image to same data type to that of Background frame i.e. double
 CurrentFrame = double(Img);
 
 % extract moving object 
 [xc,yc,stats,indexOfMax,flag] = movobj(CurrentFrame,Background);
 % goes to th function [xc,yc,stats,indexOfMax,flag] =extractball(CurrentFrame,Background)
 
 if flag==0
 % next iteration as no update is done
 continue
 end
 
 % plot bounding box(red) around the centre of detected moving object 
 hold on
 
rectangle('Position',[stats(indexOfMax).BoundingBox(1),stats(indexOfMax).BoundingBox(2),...
 stats(indexOfMax).BoundingBox(3),stats(indexOfMax).BoundingBox(4)], 'EdgeColor','r','LineWidth',2 ) ;
 
% For each time step, Kalman filter first makes a prediction i.e. xp
 if kinit==0
% at start set the position of prediction to centre of the frame
 xp = [h/2,w/2,0,0]';
 else
% after setting the position the moving object position is predicted from
% it's previous state matrix x(i-1,:)
 xp=A*x(i-1,:)' + B*u;
 end
 
 kinit=1;
% predicted process co-variance matrix PP
 PP = A*P*A' + Q;
 
% update of Kalman gain
 K = PP*H'/(H*PP*H'+R);
 
% claculate the current state matrix x(i,:)
 x(i,:) = (xp + K*([xc,yc]' - H*xp))';
% recording of current state position for the plot in figure
 pxc = x(i,1);
 pyc = x(i,2);
% update process co-variance matrix
 P = (eye(4)-K*H)*PP;
 
 hold on
 
% plot bounding box(green) around the centre of estimated position of moving object
rectangle('Position',[ pxc,pyc,stats(indexOfMax).BoundingBox(3),...
stats(indexOfMax).BoundingBox(4)],'EdgeColor','g','LineWidth',2 ) ; 
 
% to pause the frame by a unit of 0.1 for easier analysis
%  pause(0.3)
 
% record positions without estimations
 xpos(i)=xc;
 ypos(i)=yc;
% record positions without estimations
 pxpos(i)=pxc;
 pypos(i)=pyc;
end
%% plot position of moving object and its predicted positions in the same figure 
figure
plot(xpos,ypos,'r--')
hold on 
plot(pxpos,pypos,'g--')
title('moving object path');
xlabel('x-coordinates of moving object');
ylabel('y-coordinates of moving object'); 
hold off
 
%% function to perform background subtraction and to return position of detected object
% function name with parameters 'CurrentFrame' and 'Background' .
function [xc,yc,stats,indexOfMax,flag] = movobj(CurrentFrame,Background)
 
 xc = 0;
 yc = 0;
 flag = 0;
 indexOfMax=0;
 stats=0;
 
% subtract background & select pixels with a big difference i.e. back...
% ground subtraction with thresholding.
% ... represents continuation of line
 Imdiff = (abs(CurrentFrame(:,:,1)-Background(:,:,1)) > 10) ...
 | (abs(CurrentFrame(:,:,2) - Background(:,:,2)) > 10) ...
 | (abs(CurrentFrame(:,:,3) - Background(:,:,3)) > 10); 
% In above equation, difference is found out by subtracting the R,G,B
% channels seperately and then are threshold it and then by performing Or operation
% Morphology Operation erode to remove small noise applied twice
 Imdiff = bwmorph(Imdiff,'erode',2);
 Imdiff = imfill(Imdiff,'holes');
 
% select largest object
% below line returns measurements for the set of properties for each...
% 8-connected component (object) in the binary image 'Imdiff'
 stats = regionprops(Imdiff,'basic');
 [N,W] = size(stats);
 if N < 1
% return initialized value if size of stats is too small.
 return 
 end
% below line returns the maximum area of the largest moving object and its index .
 [maxArea, indexOfMax] = max([stats.Area]);
 
% specify of 'basic' = regionprops computes only the 'Area', 'Centroid', and 'BoundingBox' measurements.
if maxArea < 100 
% if are is less than 100 returns the initialized value 
 return 
 end
 
% get centre of mass and radius of largest moving object obtained using regionprops on image)
 centroid = stats(indexOfMax).Centroid ;
 
% above lines gives a row matrix (1 x 2) containing co-ordinates of detected object.
 xc = centroid(1);
 yc = centroid(2);
 
 %To indicate the updated value of co-ordinates and radius, flag is set to 1.
 flag = 1;
 
 % return updated values
 return
 
end