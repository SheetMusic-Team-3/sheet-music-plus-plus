# dictionary of key signatures
keySigSemanticToLP = {
    "Cm":   "\\key c \\minor ",
    "CM":   "\\key c \\major ",
    "C#m":  "\\key cis \\minor ",
    "C#M":  "\\key cis \\major ",
    "Cbm":  "\\key ces \\minor ",
    "CbM":  "\\key ces \\major ",
    "Dm":   "\\key d \\minor ",
    "DM":   "\\key d \\major ",
    "D#m":  "\\key dis \\minor ",
    "D#M":  "\\key dis \\major ",
    "Dbm":  "\\key des \\minor ",
    "DbM":  "\\key des \\major ",
    "Em":   "\\key e \\minor ",
    "EM":   "\\key e \\major ",
    "E#m":  "\\key eis \\minor ",
    "E#M":  "\\key eis \\major ",
    "Ebm":  "\\key ees \\minor ",
    "EbM":  "\\key ees \\major ",
    "Fm":   "\\key f \\minor ",
    "FM":   "\\key f \\major ",
    "F#m":  "\\key fis \\minor ",
    "F#M":  "\\key fis \\major ",
    "Fbm":  "\\key fes \\minor ",
    "FbM":  "\\key fes \\major ",
    "Gm":   "\\key g \\minor ",
    "GM":   "\\key g \\major ",
    "G#m":  "\\key gis \\minor ",
    "G#M":  "\\key gis \\major ",
    "Gbm":  "\\key ges \\minor ",
    "GbM":  "\\key ges \\major ",
    "Am":   "\\key a \\minor ",
    "AM":   "\\key a \\major ",
    "A#m":  "\\key ais \\minor ",
    "A#M":  "\\key ais \\major ",
    "Abm":  "\\key aes \\minor ",
    "AbM":  "\\key aes \\major ",
    "Bm":   "\\key b \\minor ",
    "BM":   "\\key b \\major ",
    "B#m":  "\\key bis \\minor ",
    "B#M":  "\\key bis \\major ",
    "Bbm":  "\\key bes \\minor ",
    "BbM":  "\\key bes \\major "
}

# dictionary of note lengths
# need to add double dots
lengthToNum = {
    "quadruple_whole":          "\\longa ",
    "quadruple_whole.":         "\\longa. ",
    "double_whole":             "\\breve ",
    "double_whole.":            "\\breve. ",
    "whole":                    "1 ",
    "whole.":                   "1. ",
    "whole..":                  "1.. ",
    "half":                     "2 ",
    "half.":                    "2. ",
    "half..":                   "2.. ",
    "quarter":                  "4 ",
    "quarter.":                 "4. ",
    "quarter..":                "4. ",
    "eighth":                   "8 ",
    "eighth.":                  "8. ",
    "eighth..":                 "8.. ",
    "sixteenth":                "16 ",
    "sixteenth.":               "16. ",
    "sixteenth..":              "16.. ",
    "thirty_second":            "32 ",
    "thirty_second.":           "32. ",
    "thirty_second..":          "32.. ",
    "sixty_fourth":             "64 ",
    "sixty_fourth.":            "64. ",
    "sixty_fourth.,":           "64.. ",
    "hundred_twenty_eighth":    "128 "
}

# dictionary of note pitches
letterToNote = {
    "C":   "c ",
    "C#":  "cis ",
    "Cb":  "ces ",
    "D":   "d ",
    "D#":  "dis ",
    "Db":  "des ",
    "E":   "e ",
    "E#":  "eis ",
    "Eb":  "ees ",
    "F":   "f ",
    "F#":  "fis ",
    "Fb":  "fes ",
    "G":   "g ",
    "G#":  "gis ",
    "Gb":  "ges ",
    "A":   "a ",
    "A#":  "ais ",
    "Ab":  "aes ",
    "B":   "b ",
    "B#":  "bis ",
    "Bb":  "bes "
}

# dictionary of clefs
clefDict = {
    "C1":   "soprano ",
    "C2":   "mezzosoprano ",
    "C3":   "alto ",
    "C4":   "tenor ",
    "C5":   "baritone ",
    "F3":   "baritone ",
    "F4":   "bass ",
    "F5":   "subbass ",
    "G1":   "french ",
    "G2":   "treble "
}

BEATS_PER_MEASURE = "1"


def parser(toparse):
    """ parses each command output of the semantic model,
        returns the equivalent LilyPond command
        inputs: toparse -- TODO
        outputs: TODO
    """
    divided_string = toparse.split("-")
    global BEATS_PER_MEASURE

    if divided_string[0] == "clef":
        return "\\clef " + clefDict[divided_string[1]] + "\n"

    elif divided_string[0] == "keySignature":
        return keySigSemanticToLP[divided_string[1]] + "\n"

    elif divided_string[0] == "timeSignature":
        BEATS_PER_MEASURE = divided_string[1][0]
        if divided_string[1] == "C":
            return "\\time 4/4 \n"
        elif divided_string[1] == "C/":
            return "\\time 2/2 \n"
        else:
            return "\\time " + divided_string[1] + "\n"

    # this is the number of beats, so somehow we need to keep track of the time signature
    elif divided_string[0] == "multirest":
        if BEATS_PER_MEASURE != 0:
            return "\\compressFullBarRests \n \t R1*" +\
                str(int(divided_string[1])//int(BEATS_PER_MEASURE)) + "\n "
        else:
            return "\\compressFullBarRests \n \t R1*" +\
                divided_string[1] + "\n "

    elif divided_string[0] == "barline":
        return " \\bar \"|\" "
        # return " "

    elif divided_string[0] == "rest":
        ending = ""
        if divided_string[1][-7:] == "fermata":
            ending = "\\fermata "
            divided_string[1] = divided_string[1][:-8]
        return "r" + lengthToNum[divided_string[1]] + ending + " "

    elif divided_string[0] == "note":
        note_info = divided_string[1].split("_", 1)
        ending = ""
        if note_info[1][-7:] == "fermata":
            ending = "\\fermata "
            note_info[1] = note_info[1][:-8]
        if int(note_info[0][-1]) == 3:
            return letterToNote[note_info[0][:-1]] +\
                lengthToNum[note_info[1]] + ending
        elif int(note_info[0][-1]) < 3:
            return letterToNote[note_info[0][:-1]] +\
                (3 - int(note_info[0][-1])) * "," +\
                lengthToNum[note_info[1]] + ending
        elif int(note_info[0][-1]) > 3:
            return letterToNote[note_info[0][:-1]] +\
                (int(note_info[0][-1]) - 3) * "\'" +\
                lengthToNum[note_info[1]] + ending

    elif divided_string[0] == "gracenote":
        note_info = divided_string[1].split("_", 1)
        if int(note_info[0][-1]) == 3:
            notepart = letterToNote[note_info[0][:-1]] +\
                lengthToNum[note_info[1]]
        elif int(note_info[0][-1]) < 3:
            notepart = letterToNote[note_info[0][:-1]] +\
                (3 - int(note_info[0][-1])) * "," + lengthToNum[note_info[1]]
        elif int(note_info[0][-1]) > 3:
            notepart = letterToNote[note_info[0][:-1]] +\
                (int(note_info[0][-1]) - 3) * "\'" + lengthToNum[note_info[1]]
        return "\\grace { " + notepart + " }"

    elif divided_string[0] == "tie":
        return "~"

    else:
        return "% could not find a match for the musical element" + toparse


def generate_music(model_output, piece_title):
    """ calls the parser to parse the input,
        sends it to a LilyPond file to generate a PDF
        inputs: model_output -- string output produced by the semantic model
                piece_title -- title of the piece, which becomes the filename
        outputs: TODO
    """
    element_list = model_output.split("\n")

    lilyPond = "\\version \"2.20.0\" \n\header{\n  title = \"" +\
        piece_title + "\"\n}\n\\score{ {\n"

    # f = open(piece_title + ".ly", "w+")
    # f.write("\\version \"2.20.0\" \n\header{\n  title = \"" + piece_title + "\"\n}\n\\score{ {\n")

    for x in range(len(element_list)-1):
        next_elem = parser(element_list[x])
        print(next_elem)
        lilyPond += next_elem
        # f.write(parser(element_list[x]))

    lilyPond += "} \n \\midi{} \n \\layout{} }"
    return lilyPond
    # f.write("\\midi{} \n \\layout{} }")
    # f.close()
