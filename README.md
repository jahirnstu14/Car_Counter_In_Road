# Car_Counter_In_Road

Welcome to the **Car_Counter_In_Road** project! This project demonstrates how to detect and count cars on a road using the YOLO (You Only Look Once) object detection model. The model is integrated with OpenCV to process video feeds, making it capable of counting vehicles in real time.

## Table of Contents

-   [Project Overview](#project-overview)
-   [Technologies Used](#technologies-used)
-   [Installation](#installation)
-   [Usage](#usage)
-   [Result](#result)

## Project Overview

This project provides a system that counts the number of cars passing through a road using the YOLO object detection model. YOLO is chosen for its speed and accuracy in detecting objects in real time. The project processes video footage from road cameras, identifies cars, and keeps a count of them. The interface is built using Python and OpenCV, making the solution suitable for real-time applications.

## Technologies Used

-   **YOLO**: For detecting cars in the video feed.
-   **OpenCV**: For handling video processing, frame extraction, and drawing bounding boxes.
-   **NumPy**: For numerical operations.
-   **Python**: The main programming language for developing the application.

## Installation

To get started with this project, follow these steps:

### 1. **Clone the Repository**

Clone the repository to your local machine:

`git clone https://github.com/your-username/car-counter-on-road.git
cd car-counter-on-road` 

### 2. **Set Up a Virtual Environment**

Creating a virtual environment ensures that the dependencies are isolated from your global Python environment. Follow these steps:

#### On Windows:

`python -m venv car_counter_env
car_counter_env\Scripts\activate` 


### 3. **Install Dependencies**

Install the required Python packages by running:

`pip install -r requirements.txt` 



## Usage

### 1. **Run the Python Script**

Once everything is set up, you can start counting cars by running the following command:

`python car_counter.py` 

The script will process the video file provided in the code and display the car count in real-time.

### 2. **OpenCV GUI**

The system opens a window displaying the processed video frames, highlighting the cars with bounding boxes. The count of detected cars is displayed on the screen, and the output is updated frame by frame as cars pass by.


## Result

Once the application is running, you will see the video feed with detected cars highlighted by bounding boxes. The total number of cars counted will also be displayed.
![First : ](https://github.com/jahirnstu14/Car_Counter_In_Road/blob/main/screenshot1.jpg)
![Second : ](https://github.com/jahirnstu14/Car_Counter_In_Road/blob/main/screenshot2.jpg)
