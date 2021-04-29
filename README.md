# SheetMusic++
<img src="static/graphics/logo.png" alt="SheetMusic++ logo" width="15%"/>

SheetMusic++ is a web application that allows users to upload an image of sheet music for digitization into three separate formats: PDF, LilyPond, and MIDI. The back-end is comprised of two sequential neural networks that segment and detect individual notes, which are then parsed into these outputs (described in the [Usage](##Usage) section). This allows musicians to easily transcribe, edit, and listen to their sheet music.

## About
This app was created in 2021 by Kian Chamine, James Karsten, Hilary Nelson, and Jack Weiler, a group of students at Pitzer College enrolled in Software Development at Harvey Mudd College. A detailed report that describes the motivations, functional requirements, architecture, algorithms, testing, and limitations can be found [here]().

## Usage
This app is deployed with Python Anywhere and can be accessed via the following link: [www.sheetmusicplusplus.com](https://www.sheetmusicplusplus.com). Note that in order to reduce costs, the AWS endpoints are currently shut down, so users cannot currently access 

The app is run using the following process:
1. Use the file browser to select a JPG or PNG image.
1. Click the "Analyze my sheet music!" button.
1. Add a title to the piece and confirm that it is readable and oriented correctly.
1. Click the "Confirm" button.
1. When the process is complete, click any of the "Download" buttons to download the following file types:
    1. LilyPond: This is a digital music composition markdown format (similar to LaTeX) that can be edited using any plain text editor. To compile it into sheet music, either [download LilyPond](http://lilypond.org/download.html) and install it on your personal machine or copy and paste the code into an online editor such as [LilyBin](http://lilybin.com) or [HackLily](https://hacklily.org).
    1. PDF: This is a highly compatible document that can be read on most machines and physically printed. It has a small file size that allows for easy storage and sharing.
    1. MIDI: This stands for Musical Instrument Digital Interface, which is a widely-used, editable music format that stores complex notation in a small file size. Both the notes and overall information such as tempo can be adjusted, and the track can be rendered using thousands of synthesized and sampled instruments. Programs such as [GarageBand](https://www.apple.com/mac/garageband) and [Logic Pro](https://www.apple.com/logic-pro) can edit and export MIDI tracks into various audio formats such as MP3 or WAV. There are also free online compilers such as [signal](https://signal.vercel.app) and [SolMiRe](https://solmire.com/midieditor) that can edit basic features of MIDI tracks. Read more [here](https://blog.landr.com/what-is-midi).
1. In order to restart the process with a different piece of music, click the "Analyze another piece of sheet music" button.

## Known Bugs & Limitations
This app is in a V1 release, so all major known bugs have been resolved. However, there are some limitations worth noting:
1. Any images that are oriented incorrectly, crooked, or warped often are unable to be read, which prompts the user to upload a new image. It is recommended to upload either a scanned image or a high-quality image taken in good lighting for optimal results.
1. Overall, the OMR neural network has foundational issues, including an incomplete dictionary, insufficient training on imperfect images, and an inability to seamlessly combine different line of music. These prevent it from making highly accurate predictions, so in most cases, it is necessary for the user to edit and re-compile the LilyPond file to fix errors. Future work on this project would likely require retraining the model with a new dictionary or implementing a new model entirely.

We invite users to report any new issues [here](https://github.com/SheetMusic-Team-3/MVP/issues).

## Credits
It utilizes a [You Only Look Once: Unified, Real-Time Object Detection](https://arxiv.org/abs/1506.02640) model, developed in 2016 by Joseph Redmon, Santosh Divvala, Ross Girshick, and Ali Farhadi, to segment the image into lines. It utilizes a [End-to-End Neural Optical Music Recognition of Monophonic Scores](https://www.mdpi.com/2076-3417/8/4/606/htm) model, developed in 2018 Jorge Calvo-Zaragoza and David Rizo, to perform the note recognition. The web app is based off the open-source code [web-omr](https://github.com/liuhh02/web-omr), published in 2019 by liuhh02.

## License
The source code for the site is licensed under the MIT license, which you can find in the MIT-LICENSE.txt file.
All graphical assets are licensed under the Creative Commons Attribution 3.0 Unported License.