# SheetMusic++ (MVP)
SheetMusic++ is a web app that allows a user to upload an image of a piece of sheet music and outputs an editable LilyPond file of that music.

## Usage
This app is deployed with Python Anywhere and can be accessed via the following link: [www.sheetmusicplusplus.com](https://www.sheetmusicplusplus.com).

The app is run using the following process:
1. Use the file browser to slect a .JPG or .PNG image
1. Click the "Analyze my sheet music!" button
1. When the process is complete, click "Download LilyPond file"
1. This file can be edited using any plain text editor. To compile it into sheet music, either [download LilyPond](http://lilypond.org/download.html) and install it on your personal machine or copy and paste the code into an online editor such as [LilyBin](http://lilybin.com) or [HackLily](https://hacklily.org)

## Known Bugs
(Many)

We invite users to report any new issues [here](https://github.com/SheetMusic-Team-3/MVP/issues).

## Credits
This app was created in 2021 by Kian Chamine, James Karsten, Hilary Nelson, and Jack Weiler, a group of Pitzer College students. It utilizes a [You Only Look Once: Unified, Real-Time Object Detection](https://arxiv.org/abs/1506.02640) model, devleoped in 2016 by Joseph Redmon, Santosh Divvala, Ross Girshick, and Ali Farhadi, to segemnt the image into lines. It utilizes a [End-to-End Neural Optical Music Recognition of Monophonic Scores](https://www.mdpi.com/2076-3417/8/4/606/htm) model, developed in 2018 Jorge Calvo-Zaragoza and David Rizo, to perform the note recognition. The web app is based off the open-source code [web-omr](https://github.com/liuhh02/web-omr), published in 2019 by liuhh02.
