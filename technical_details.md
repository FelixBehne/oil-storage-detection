# Business Case
Aboveground Petroleum, Oil and Lubricant (POL) storage areas are common in manufacturing and government facilities. They fall under critical infrastructure for transportation (e.g. vehicles, ships and aircraft) and manufacturing industries (e.g. refineries, power stations, manufacturing). 
 
Deep Learning can automatically detect the number, size, and type of POL storage present on a site. This can help monitor the state of aboveground fuel storage tanks, including preventing spills, overfills, and corrosion.

The following business use cases can be identified: 

* Detect POL storage area to subsequently monitor the site for safety or environmental issues (e.g. fire hazard)
* Countries try to keep their oil reserves and production a secret. However, oil prices have a significant influence on the prices of all commodities. POL storage area detection would allow higher transparency leading to a potential competitive advantage. 
* Military: Assess critical infrastructure for strategic insight.

There are different companies that already target these business cases: 
* [Planet](https://www.planet.com)
* [Orbitalinsights](https://www.oribitalinsights.com)
# Dataset Overview
## Samples
![example1](0a75eb88-46ba-4a64-acb3-8919ea880137.jpg)![example2](0f9798fb-d940-4355-828b-894cc998f6fa.jpg)
## Stats
- 98 Images
- 2560x2560
- 1.2m
- Only one object class
## Particularities
- Satelite imagery
  - Always the same perspective
  - Minor size distortion whithin the images
- Very large images for object detection
- Includes very small objects
- Objects are simple compared to other tasks: i.e.: detect white/grey/black circles
# Thoughts on Implementation
- The Dataset is extremely small
  - Try using a pretrained model and transfer learning
  - Try spliting the images into smaller ones
  - Use augmentation
- Try state of the Art Object Detectors with TF support:
  - YOLOv5, YOLOX
  - EfficientDet
- Consider simpler approaches due to object simplicity:
  - Can we use the fact that all objects are somwhat circular?
- Augmentation techniques which might make sence:
  - Horizontal and vertical flip
  - Rotate
  - Color / brightness augmentations to simulate different environments and weather conditions