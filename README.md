# SheetMusic++ (MVP)
SheetMusic++ is a web app that allows a user to upload an image of a piece of sheet music and outputs an editable LilyPond file of that music.

## Usage
This app is deployed with Python Anywhere and can be accessed via the following link: [www.sheetmusicplusplus.com](https://www.sheetmusicplusplus.com).

The app is run using the following process:
1. Use the file browser to slect a .JPG or .PNG image
2. Click the "Analyze my sheet music!" button
3. When the process is complete, click "Download my LilyPond"
4. This file can be edited using any plain text editor. To compile it into sheet music, either [download LilyPond](http://lilypond.org/download.html) and install it on your personal machine or copy and paste the code into an online editor such as [LilyBin](http://lilybin.com) or [HackLily](https://hacklily.org)

## Known Bugs
This app is currently under development, meaning that it has limited functionality. However, some significant bugs that we are aware of and currently working to resolve are:
1. Since there is currently limited error handling, any invalid images cause the page to redirect to an Internal Server Error 500
2. Only 1 line of music can be included in the uploaded image; otherwise this leads to an internal server error
3. Any images for which no notes can be detected cause an internal server error
4. The output file names are not parsed, so it appears that there is a double file extension (however, this is aesthetic only and does not affect the ability of the LilyPond file to run)
5. If the results page is accessed via URL without uploading an image first, an internal server error will occur
6. Overall, the neural network has foundational issues that prevent it from making highly accurate predictions

We invite users to report any new issues [here](https://github.com/SheetMusic-Team-3/MVP/issues).

## Credits
This app was created in 2021 by Kian Chamine, James Karsten, Hilary Nelson, and Jack Weiler, a group of Pitzer College students. It utilizes a [You Only Look Once: Unified, Real-Time Object Detection](https://arxiv.org/abs/1506.02640) model, devleoped in 2016 by Joseph Redmon, Santosh Divvala, Ross Girshick, and Ali Farhadi, to segemnt the image into lines. It utilizes a [End-to-End Neural Optical Music Recognition of Monophonic Scores](https://www.mdpi.com/2076-3417/8/4/606/htm) model, developed in 2018 Jorge Calvo-Zaragoza and David Rizo, to perform the note recognition. The web app is based off the open-source code [web-omr](https://github.com/liuhh02/web-omr), published in 2019 by liuhh02.
