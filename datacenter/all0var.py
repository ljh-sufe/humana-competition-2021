
all0var = []
file = open(r"C:\Users\41409\Documents\wustlCourses\business_competition\dataCenter\all0var.csv")
for x in file.readlines():
    all0var.append(x.split(" ")[0])