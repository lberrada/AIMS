""" Description here

Author: Leonard Berrada
Date: 24 Oct 2015
"""

import csv

def format_name(e):
    
    res = e.replace(" ", "")
    if "+" in res:
        list_of_names = res.split("+")
        for k in range(len(list_of_names)):
            list_of_names[k] = "".join([name[0].upper() for name in list_of_names[k].split("_")])
        res = "+".join(list_of_names)
    if "*" in res:
        list_of_names = res.split("*")
        for k in range(len(list_of_names)):
            list_of_names[k] = "".join([name[0].upper() for name in list_of_names[k].split("_")])
        res = "$\\times$".join(list_of_names)
    res = res.replace("EQ2", "EQ")
    res = res.replace("constant", "C")
    res = res.replace("temperature", "TP")
    
    return "\\multirow{2}{*}{%s}" % res

def format_var(e):
    
    if e in ["sigma_n", "sigma_f", "nu"]:
        return "$\\" + e + "$"
    
    if e == "p_period":
        return "$\\rho$"
    
    if e == "c_alpha":
        return "$\\alpha$"
    
    if e == "p_scale":
        return "$\\sigma_p$"
    
    if e == "eq_sigma_f":
        return "$\\sigma_{f,EQ}$"
    
    if e == "rq_scale":
        return "$\\l_{RQ}$"
    
    if e == "rq_sigma_f":
        return "$\\sigma_{f,RQ}$"
    
    if e == "p_sigma_f":
        return "$\\sigma_{f,P}$"
    
    if e == "eq_scale":
        return "$l_{EQ}$"
    
    if e == "rq_nu":
        return "$\\nu_{RQ}$"
    
    
def format_nb(e):
    
    return "% 3.2G" % float(e)

def format_score(e):
    
    return "\\multirow{2}{*}{% 3.3G}" % float(e)
    

for k in range(4):
    file_to_parse = './out/results_v3-' + str(k) + '.csv'
    
    with open(file_to_parse, 'r', newline='') as csvfile:
            my_reader = csv.reader(csvfile, delimiter='\t',
                                   quoting=csv.QUOTE_MINIMAL)
            contents = [row for row in my_reader]
            
    with open(file_to_parse[:-4] + "new.csv", 'w', newline='') as csvfile:
            my_writer = csv.writer(csvfile, delimiter='&',
                                   quoting=csv.QUOTE_MINIMAL)
            flag = -1
            contents_to_write = []
            scores = []
            for row in contents:
                if row[0][0] == "-":
                    flag = 0
                    continue
                
                if flag == 0:
                    row_to_write = [format_name(e) for e in row[:2]]
                    flag += 1
                    continue
                
                if flag == 1:
                    row_to_write += [format_var(e) for e in row] + [""]
                    flag += 1
                    contents_to_write.append(row_to_write)
                    continue
                
                if flag == 2:
                    row_to_write = [''] * 2 + [format_nb(e) for e in row]
                    flag += 1
                    continue
                    
                if flag == 3:
                    scores.append(format_score(row[1]))
                    row_to_write.append("\\midrule")
                    contents_to_write.append(row_to_write)
                    row_to_write = []
                    flag = -1
                    continue
                
            n = max([len(row) for row in contents_to_write])
            my_writer.writerow(["\\begin{table}[H]"])
            my_writer.writerow(["\\small"])
            my_writer.writerow(["\\centering"])
            my_writer.writerow(["\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill} } lc|" + "c"*(n - 3) + "|r}"])
            my_writer.writerow(["\\toprule"])
            my_writer.writerow(["Kernel", "Mean", "\\multicolumn{%i}{c}{Hyperparameters}"%(n-4), "Score \\\\ \\midrule"])
            for k in range(len(contents_to_write)):
                aux = contents_to_write[k]
                contents_to_write[k] = aux[:-1] + [''] * (n - len(contents_to_write[k]))
                if "midrule" not in aux[-1]:
                    contents_to_write[k] += [scores[k // 2] + aux[-1] + "\\"*2]
                else:
                    contents_to_write[k] += [aux[-1].replace("\\midrule", "") + "\\"*2 + " \\midrule"]
            contents_to_write[-1] = contents_to_write[-1][:-1] + [contents_to_write[-1][-1].replace("mid", "bottom")]
            [my_writer.writerow(row) for row in contents_to_write]
            my_writer.writerow(["\\end{tabular*}"])
            my_writer.writerow(["\\caption{...}"])
            my_writer.writerow(["\\label{label}"])
            my_writer.writerow(["\\end{table}"])
            
