import subprocess

def test( ckpt_path, dataset ):
    out = subprocess.check_output([ "python3", "test.py", "--ckpt_path", ckpt_path, "--dataset", dataset ])

    out = out.decode('utf-8')
    out = out.split('\n')
    metrics = None
    for s in out:
        if '#Total#' in s:
            metrics = s
    print( metrics )
    metrics = metrics.split('  ')
    auc = metrics[2].split(':')[1]
    f1 = metrics[3].split(':')[1]
    pr_auc = metrics[4].split(':')[1]
    err = metrics[5].split(':')[1]

    print( auc, f1, pr_auc, err )
    return [ auc, pr_auc ]

def main():
    header = [ "" ,"DFDC" ]
    m = [ "", "AUC", "PR AUC" ]
    rows = []
    ckpt = {
        "crosshash" : "./ckpts/crosshash.pth.tar",
        "DCL"       : "./ckpts/DCL.pth.tar",
        "DCL+jpeg"  : "./ckpts/DCL_jpeg.pth.tar",
        "DCL+Hash"  : "./ckpts/DCL_hash.pth.tar",
    }

    for method, ckpt_path in ckpt.items():
        print(f"Testing {method}...")
        ret = test( ckpt_path, "ffpp" )
        rows.append( [ method ] + ret )

    print( rows )

    import csv
    with open( './result.csv', 'a', encoding = 'UTF8', newline = '' ) as f:
        writer = csv.writer(f)
        # write the header
        writer.writerow("")
        writer.writerow(header)
        writer.writerow(m)

        # write the data
        writer.writerows(rows)
if __name__ == '__main__':
    main()

