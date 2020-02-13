# Project files and  directories  structure​
-   Root​    
	-   Aggregate_products.py​    
	-   Get_product_data.py​    
	-   aggregate_structured_wls.py​    
	-   neo4j_graph.py​    
	-   create_neo4j_from_csv.py​    
	-   Aggregating​    
		-   Aggregate_*.py (pep, obr, lei, crb)​    
		-   dynamodb_edge_rank.py​    
		-   generic_aggregation.py​    
		-   dynamodb_populate.py
	-   Utils​    
		- utils.py
# Data files and  directories  structure​

-   Root​
	- dynamodb_wls_names.json​    
	- Product (CRB, OBR, LEI or PEP):​    
		- *_data.pkl (Dictionary): SQL db product entities data;​    
		-   *_relationships.pkl  (Dictionary): SQL db product relationships data;​    
		-   Flagged_*.csv: Product entities matched in WLS. Will be used to populate DynamoDB;​    
		-   neo4j_edges.csv, neo4j_nodes.csv: csv files in the format to create a new Graph DB on Neo4j.​    
	-   Structured_wls:​    
		-   wls_data.pkl (Dictionary): SQL db product entities data;​    
		-   id_to_names.pkl (Dictionary): SQL data formatted to be processed;​    
		-   neo4j_edges.csv, neo4j_nodes.csv: csv files in the format to create a new Graph DB on Neo4j.​  ​

## aggregate_products.py​

-   Match WLS names  from  the  table "ISS_WLS_NAMES" with  names  from  other  products​    
	-   Right  now  can match with: OBR, LEI, CRB and PEP​
-   Flags:​    
	-   --root_path (required): The path where all the data to be processed is. All the data need to be in a specific format and structure. If nothing is found, it will query the data in save it in the format and structure needed;​
    -   --product (optional): Which product to process, default is "all";​
    -   --flagged_file (optional): Where the file with the DynamoDB wls names scanned table is. If not given will scan the table and save the result in the root path.​

## get_product_data.py​

-   Get product data from SQL data and save in the format to be processed in the aggregation functions.​    
-   Flags:​    
	-   --product (required): Which product to fetch data;​    
		-   As of now: CRB, LEI, OBR or PEP​    
	-   --root_path (required): Path to the root directory where the data will be saved

## neo4j_graph.py​

-   Process a product (PEP, OBR, LEI, CRB) to create a .csv file in the neo4j format,​  
    so that it can create a neo4j graph.​
-   Flags:​    
	-   --root_path (required): The root path of the data directory, where the script will read the dictionary data and save the neo4j .csv files;​
    -   --product (required): Which product to process.

## create_neo4j_from_csv.py​

-   Create a Neo4j graph using "neo4j-admin import" and the .csv files created by "neo4j_graph.py"​    
-   Flags:​    
	-   -r, --root_path (required): The root path of the data directory, where the script will read all neo4j .csv files;​    
	-   -n, -neo4j_file_path (required): Path to neo4j-admin file;​    
	-   -d, --db_name (required): Graph database name
	
## aggregating/aggregate_*.py (PEP, OBR, LEI, CRB)

-   If not present, fetch data from SQL product table and transform to the format used in the process.​
-   Match names from the product with names from WLS;​    
-   This scripts are used in aggregate.py

## aggregate_structured_wls.py​

-   Get SQL data and save in Dictionary format to create neo4j .csv file​    
-   Create neo4j .csv file​    
-   Flags:​    
	-   --root_path (required): The root path where the data will be processed and saved;​   
	-   --get_data (optional): If set to 'True" the script will fetch SQL data and save the results after processing in the root path.
## aggregating/generic_aggregation.py​

- Generic functions used in aggregate_*.py​
## aggregating/dynamodb_populate.py​
-   Populate DynamoDB table with the names flagged using aggregate_*.py​    
-   Flags:​	    
	-   --root_path (required): The root path with all products where the flagged_*.csv files are saved;​	    
	-   --product (optional): Which products to populate DynamoDB table. Default is "all" (PEP, OBR, LEI, CRB).​
## aggregating/dynamodb_edge_rank.py​
- Using .csv file created by Neo4j with the edges rank property populate the DynamoDB entities links with their rank
## utils/utils.py​

- Generic utils functions.​


# Neo4j​

## Creating  Graph​
-   Fetch SQL data using get_product_data.py script​    
-   Transform SQL data in .csv file in the neo4j-admin graph creation format​    
	>   [https://neo4j.com/docs/operations-manual/current/tools/import/file-header-](https://neo4j.com/docs/operations-manual/current/tools/import/file-header-format/)[format/](https://neo4j.com/docs/operations-manual/current/tools/import/file-header-format/)​
	>   Neo4j-admin is way faster than using neo4j IMPORT CSV method​    
	-   Nodes: ​    
		-   Required: ":ID"​    
		-   Optional: ":LABEL", node_properties​    
	-   Edges: ​    
		-   Required: ":START_ID", ":TYPE", ":END_ID"​    
		-   Optional: edge_properties​    
	>   To transform the data in the .csv format needed, use neo4j_graph.py script
-   Load the scripts in the neo4j-admin import​    
	>   [https://neo4j.com/docs/operations-manual/current/tools/import/syntax/](https://neo4j.com/docs/operations-manual/current/tools/import/syntax/)​
	>   For that there is a python script named "create_neo4j_from_csv.py"

## Using neo4j
-   There are two options:​    
	-   Neo4j desktop​    
	-   Neo4j cypher-shell​    
-   Link to download:​    
	>   [https://neo4j.com/download/](https://neo4j.com/download/)​
	
-   Download both neo4j desktop and neo4j server
-   Using neo4j desktop is easier to install plugins, to install plugin in neo4j server there are dependencies to install.​    
-   Using neo4j server is faster to load data and transform in graph​    
	>   Neo4j-admin import takes seconds with a few million nodes and edges, cypher LOAD CSV command can take hours and crash.​    

-   The best way I found to use neo4j:​    
	-   Create a graph using neo4j-admin import​    
	-   Export that graph to neo4j desktop​    
	-   Use neo4j desktop with plugins installed​    
	-   Execute complex queries in cypher-shell and visualize results in neo4j desktop

## Important configuration​
-   apoc.export.file.enabled=true​    
	-   Enables export of .csv files with queried results, useful in getting edge rank​    
-   dbms.security.procedures.unrestricted=apoc.*,algo.*​    
	-   Allow apoc and algo methods from plugins​    
-   dbms.memory.heap.max_size=5G​    
	-   Some queries need more memory, default is 1GB​    

## Setp-by-step

-   Fetch sql data, transform, and save in the format used in the project​    
	>  For PEP, CRB, LEI and OBR use get_product_data.py​    
	>  For structured wls use aggregate_structured_wls.py​    
	
-   Create neo4j .csv files using "neo4j_graph.py" script​    
-   Create neo4j graph using "create_neo4j_from_csv.py"​    
-   Export the created graph to neo4j desktop​    
	> Copy the directory created in the data/database of the Neo4j server directory to the data/database directory of the Neo4j desktop graph created
	E.g.: /home/lucas/neo4j/neo4j-community-3.5.13/data/databases/from_scratch_graph -> /home/lucas/.config/Neo4j Desktop/Application/neo4jDatabases/database-0f9e0458-b61c-4ee6-8f60-64fdd89c448c/installation-3.5.6/data/databases/graph.db

	- It's possible to find the Neo4j desktop graph directory in the "Open Folder" button ![neo4j image]([https://i.imgur.com/6KYs1S7.png](https://i.imgur.com/6KYs1S7.png))

-   Install plugins and modify graph configuration
