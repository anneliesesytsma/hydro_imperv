This code documents the process used to develop the HCIA ArcGIS tool. The tool was developed in ArcGIS 10.7 and requires the Spatial Analyst extension and ArcHydro tools. ArcHydro is available for download at the following link: http://downloads.esri.com/archydro/archydro/

# Preparing ArcMap for HCIA Toolbox Use 

1.	File structure
  
    a.	Create a new folder to house inputs and outputs
    
    b.	Create a new file geodatabase to hold input data (‘In.gbd’)
    
    c.	Create a new file geodatabase to hold output data (‘Out.gbd’)
  
2.	Set default workspace
    
    a.	Geoprocessing -> Environments -> Workspace
    
    b.	Set Current Workspace to your ‘Out.gbd’ path
    
3.	Pathnames
    
    a.	File -> Map Document Properties 
    
    b.	Select ‘store relative pathnames to data sources’


# The following input data are required for using the HCIA toolbox:

1.	Digital elevation model (DEM) - 	Raster - Sufficient resolution for urban areas (<= 1 m recommended)

2.	Impervious surfaces	- Polygon - Requires a field designating each polygon into one of three impervious classes: (1) Roof, (2) Roads, (3) Other.

3.	Area of interest - Polygon - Can be a watershed or city boundary, etc.

4.	Drainage points -	Point	 - Manholes, catch basins, etc.

5.	Roof area connectivity - Polygon - A field designating the initial guess of rooftop connectivity.  


# Limitations

The HCIA ArcGIS tool was developed for a finite number of scenarios. These parameters for which the HIA ArcGIS tool is accurate are summarized in model documentation. For additional background on the develpment of the ArcGIS tool, see model documentation here: 
