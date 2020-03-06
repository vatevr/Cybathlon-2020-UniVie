def artifact_removal(self, csp, info, layout):
    print("This CLI will help you remove features that look like artifacts.")
    print("Please enter an integer number (starting from 0) for each of the patterns you would like to drop.")
    print("Once you are done, enter a negative integer")
    drops = []
    while 1:
        csp.plot_patterns(info, layout=layout)
        print("Please select which features to drop [band/component]: ")
        drop = int(input())
        if drop < 0:
            return drops
        else:
            drops.append(drop)