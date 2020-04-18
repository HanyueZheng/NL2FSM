import re

filename = "segmentation.txt"
with open(filename, 'r' , encoding="utf-8") as f:
    lines = f.read()
    lines = lines.replace("</phrase> <phrase>", " ")
    # file = open("seg.txt", "w+")
    # print(lines, file=file)
    # file.close()
    iefile = open("ie_out.txt", "w+")
    file1 = open("input.txt", "r+" , encoding="utf-8")
    outlines = file1.read()
    print("outlines:" + outlines)
    linesplit = outlines.split(".")
    iearr = []
    for l in linesplit:
        if l.find(":") != -1:
            if l[l.find(":")+1:] not in iearr:
                iearr.append(l[l.find(":")+1:])
            # print(l[l.find(":")+1:], file=iefile) # first sentence after colon should be recorded
        if l.lower().find("if ") != -1:
            if l[:l.lower().find("if ")] not in iearr:
                iearr.append(l[:l.lower().find("if ")])
            if l[l.lower().find("if ")+3:] not in iearr:
                iearr.append(l[l.lower().find("if ")+3:])
            #print(l[:l.lower().find("if ")], file=iefile)
            #print(l[l.lower().find("if ")+3:], file=iefile) # sentence followed by if should be recorded.
        if l.lower().find("when") != -1:
            if l[:l.lower().find("when ")] not in iearr:
                iearr.append(l[:l.lower().find("when ")])
            if l[l.lower().find("when ")+5:] not in iearr:
                iearr.append(l[l.lower().find("when ") + 5:])
            #print(l[:l.lower().find("when ")], file=iefile)
            #print(l[l.lower().find("when ") + 5:], file=iefile)  # sentence followed by when should be recorded.
        if l.lower().find("once ") != -1:
            if l[:l.lower().find("once ")] not in iearr:
                iearr.append(l[:l.lower().find("once ")])
            if l[l.lower().find("once ")+5:] not in iearr:
                iearr.append(l[l.lower().find("once ") + 5:])
            #print(l[:l.lower().find("once ")], file=iefile)
            #print(l[l.lower().find("once ") + 5:], file=iefile)  # sentence followed by once should be recorded.
        if l.lower().find("after") != -1:
            if l[:l.lower().find("after ")] not in iearr:
                iearr.append(l[:l.lower().find("after ")])
            if l[l.lower().find("after ")+6:] not in iearr:
                iearr.append(l[l.lower().find("after ") + 6:])
            #print(l[:l.lower().find("after ")], file=iefile)
            #print(l[l.lower().find("after ") + 6:], file=iefile)
        if l.lower().find("before") != -1:
            if l[:l.lower().find("before ")] not in iearr:
                iearr.append(l[:l.lower().find("before ")])
            if l[l.lower().find("before ")+6:] not in iearr:
                iearr.append(l[l.lower().find("before ") + 6:])
            #print(l[:l.lower().find("before ")], file=iefile)
            #print(l[l.lower().find("before ") + 6:], file=iefile)  # sentence followed by before/after should be recorded.
        if l.find(" mode ") != -1:
            if l.split()[l.split().index("mode")-1]+ " " +l.split()[l.split().index("mode")] not in iearr:
                iearr.append(l.split()[l.split().index("mode")-1]+ " " +l.split()[l.split().index("mode")])
            #print(l.split()[l.split().index("mode")-1]+ " " +l.split()[l.split().index("mode")], file=iefile) # collect phrases contain mode.
        if l.lower().find(" by ") != -1:
            if l[l.lower().find("by ") + 2:] not in iearr:
                iearr.append(l[l.lower().find("by ") + 2:])
            #print(l[l.lower().find("by ") + 2:], file=iefile)  # sentence followed by by.
        if l.lower().find("must be") != -1:
            if l[:l.lower().find("must be ")] not in iearr:
                iearr.append(l[:l.lower().find("must be ")])
            if l[l.lower().find("must be") + 7:] not in iearr:
                iearr.append(l[l.lower().find("must be") + 7:])
            #print(l[:l.lower().find("must be ")], file=iefile)
            #print(l[l.lower().find("must be") + 7:], file=iefile)
        if l.lower().find("should be") != -1:
            if l[:l.lower().find("should be ")] not in iearr:
                iearr.append(l[:l.lower().find("should be ")])
            if l[l.lower().find("should be") + 9:] not in iearr:
                iearr.append(l[l.lower().find("should be") + 9:])
            #print(l[:l.lower().find("should be ")], file=iefile)
            #print(l[l.lower().find("should be") + 9:], file=iefile)
        if l.lower().find("will be") != -1:
            if l[:l.lower().find("will be ")] not in iearr:
                iearr.append(l[:l.lower().find("will be ")])
            if l[l.lower().find("will be") + 7:] not in iearr:
                iearr.append(l[l.lower().find("will be") + 7:])
            #print(l[:l.lower().find("will be ")], file=iefile)
            #print(l[l.lower().find("will be") + 7:], file=iefile)
        if l.lower().find("will not be") != -1:
            if l[:l.lower().find("will not be ")] not in iearr:
                iearr.append(l[:l.lower().find("will not be ")])
            if l[l.lower().find("will not be") + 11:] not in iearr:
                iearr.append(l[l.lower().find("will not be") + 11:])
            #print(l[:l.lower().find("will not be ")], file=iefile)
            #print(l[l.lower().find("will not be") + 11:], file=iefile)  # sentence contains will be/ will not be/should be/ must be.
        for i in l.split():
            if re.match("^[A-Z]+$", i):
                if i not in iearr:
                    iearr.append(i)
                #print(i, file=iefile)  # collect all uppercase words.
    for i in iearr:
        if i != " " and i != "\n":
            #if outlines.replace("<phrase>","").replace("</phrase>", "").find(i):
            if i[0] == "\n":
                ireplace = i[0] + "<phrase>" + i[1:] + "</phrase>"
                # for j in iearr:
                #     if j.find(i)!= -1:
                #         iearr[iearr.index(j)].replace(i, ireplace)
            else:
                ireplace = "<phrase>" + i + "</phrase>"
                # for j in iearr:
                #     if j.find(i) != -1:
                #         iearr[iearr.index(j)].replace(i, ireplace)
            outlines = outlines.replace(i, ireplace)

    outlines.replace("<phrase></phrase>", " ")
    print(outlines, file=iefile)

