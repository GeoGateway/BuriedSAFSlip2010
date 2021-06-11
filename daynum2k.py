# Date to day past y2k
def daynum2k(date):
   """
   Function daynum2k: 
      date: string in format dd-Mmm-yyyy ie 13-Nov-2009
      return day past y2k start (01-Jan-2000 => 1)
      
      Note: good for years 2000-2023 (easily generalized)
   """
   yearc = [366,365,365,365,366,365,365,365,366,365,365,365,
            366,365,365,365,366,365,365,365,366,365,365,365]
   monthc = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

   day,month,year = date.split('-')
   if int(year)%4==0:
      monthc[1]=29

   monthld = 12*[None]
   monthfd = 12*[None]
   monthld[0] = monthc[0]
   monthfd[0] = 1
   for i in range(1,12):
       monthld[i] = monthld[i-1] + monthc[i]
       monthfd[i] = monthld[i]-monthc[i]+1

   mnames = ["Jan","Feb","Mar","Apr","May","Jun",
             "Jul","Aug","Sep","Oct","Nov","Dec"]
   firstday = dict(zip(mnames,monthfd))
   # firstday should return first day of named month: firstday["Feb"] => 32
   daynum = firstday[month]+int(day) - 1
   yearld = 24*[None]
   yearfd = 24*[None]
   yearld[0] = yearc[0]
   yearfd[0] = 1
   for i in range(1,24):
       yearld[i] = yearld[i-1]+yearc[i]
       yearfd[i] = yearld[i]-yearc[i]+1
   ynames=["2000","2001","2002","2003","2004","2005",
           "2006","2007","2008","2009","2010","2011",
           "2012","2013","2014","2015","2016","2017",
           "2018","2019","2020","2021","2022","2023"]
   firstyday = dict(zip(ynames,yearfd))
   daynum2k = firstyday[year]+daynum-1
   return daynum2k

def test():
    a=daynum2k("04-Jul-2009")
    b=daynum2k("04-Jun-2009")
    c=daynum2k("31-Dec-2009")
    d=daynum2k("01-Jan-2009")
    e=daynum2k("01-Jan-2010")
    diff=a-b
    yd1=daynum2k("01-Apr-2010")-daynum2k("01-Apr-2009")
    yd2=daynum2k("01-Apr-2008")-daynum2k("01-Apr-2007")
    print "04-Jul-2009", a
    print "04-Jun-2009", b
    print "diff = ",diff
    print "31-Dec-2009", c
    print "01-Jan-2009", d
    print "01-Jan-2010", e
    print "2010-2009",yd1
    print "2008-2007 (leap)",yd2

if __name__ == '__main__':
	test()

