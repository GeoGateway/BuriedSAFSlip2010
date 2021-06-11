def setpar(s,line,vdict,valtype):
   """
   Function setpar: 
      s: string to match in the line, and dictionary key
      line: single line of text considered from file
      vdict: dictionary of values keyed by strings like s,
         to be set when match occurs
      valtype: "int" for integer, "float" for floating point 
      
      effect: set value in vdict (dictionary).  
      
      Typical use:
              setpar('Ground Range Data Longitude Samples',line,af_val,'int')
   
   Note does nothing if first or second tests 
      (s in first 40 cols of line, "=" in line) do not pass.
   
   Assumes:
       input lines have variable name strings in first 40 columns
       specified variable names occur on lines containing "="
       values occur next after "=" and in the next 16 columns
   
   These appear true for UAVSAR *.ann files, based on inspection.
   
   In words:
      check for non-comment line with s in first 40 cols; make sure '=' in line.
      chop off first part up to '='
      interpret and store next 16 cols as float or int, acc. to valtype
   """
   
   if line[0] != ';' and s in line[:40] and '=' in line:
      line = line[line.index('=')+1:]
      if valtype == 'float':
         vdict[s] = float(line[:16])
      if valtype == 'int':
         vdict[s] = int(line[:16])
      if valtype == 'string':
         vdict[s] = line[:16].strip() # strip removes leading, trailing spaces
   return vdict
