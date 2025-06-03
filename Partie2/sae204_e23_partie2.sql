SET SCHEMA 'partie2';

WbImport -file=ppn.csv 
		 -table=partie2._module 
		 -delimiter=';' 
		 -header=true 
		 -mode=insert
		 ;