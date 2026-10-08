```
/**************************************************************************\
 *                                                                         *
 *          ###########   ###########   ##########    ##########           *
 *         ############  ############  ############  ############          *
 *         ##            ##            ##   ##   ##  ##        ##          *
 *         ##            ##            ##   ##   ##  ##        ##          *
 *         ###########   ####  ######  ##   ##   ##  ##    ######          *
 *          ###########  ####  #       ##   ##   ##  ##    #    #          *
 *                   ##  ##    ######  ##   ##   ##  ##    #    #          *
 *                   ##  ##    #       ##   ##   ##  ##    #    #          *
 *         ############  ##### ######  ##   ##   ##  ##### ######          *
 *         ###########    ###########  ##   ##   ##   ##########           *
 *                                                                         *
 *            S E C U R E   M O B I L E   N E T W O R K I N G              *
 *                                                                         *
\**************************************************************************/
```
# SMARTER Field Test Dataset - Branch lalmon

**This repository contains an updated dataset of the SMARTER field test**

The SMARTER dataset has been refined after the initial release.

First by Flor Álvarez for her dissertation in: https://github.com/tu-fmaz/smarter_traces_field_test

Then, further, by Lars Almon for his dissertation as presented here.

## Field Test

The dataset was gathered during the SMARTER field test.
For more details about the project Smartphone-based Communication Networks for Emergency Response (SMARTER) see the [SMARTER project homepage](https://smarter-projekt.de/).

The field test involved 125 volunteer participants using the SMARTER Android App during a scripted emergency scenario. 
For an impression of the field test, see the [field test video](https://www.youtube.com/watch?v=Hb8mgVJHrs0).

## Data

We gathered the participant movement data as well as network metrics.
The following data files are provided:
 - connections.db: Information about established network connections.
 - ids.db: ID mapping of smartphones.
 - messages.db: Exchanged network messages.
 - neighbors.db: Information about network neighbors over time.
 - realm-db: realm database of each device.

## Usage

0. Install Python 3 and the dependencies `pip install -r requirements.txt`.

1. Change the DATA_PATH variable in *config.py* to the folder containing the experiment data. With the subfolders ibr-logs, realm-db, sensor-traces and the contactlist.csv in it.

2. If you plan to run everything, maybe set AUTO_OPEN_PLOTS to False to prevent the auto opening of plots.

3. To run The One with the generated files checkout the example configuration file *smarter_settings.txt*

4. *fuse_databases.py*
	- combines all node data and the other created databases into one smarter.db for easier sharing

5. *sqlite2OneMovements.py*
	- standalone command line utility which can create a The One movement trace file from a folder of sqlite databases in which each db contains a gps trace for a node

## Authors

* **Flor Álvarez** ([email](mailto:falvarez@seemoo.tu-darmstadt.de), [web](https://www.seemoo.tu-darmstadt.de/team/falvarez/))
* **Lars Almon** ([email](mailto:lalmon@seemoo.tu-darmstadt.de), [web](https://seemoo.de/lalmon))
* **Patrick Lieser** 
* **Tobias Meuser** 
* **Yannick Dylla** 
* **Björn Richerzhagen** 
* **Matthias Hollick** 
* **Ralf Steinmetz** 

## License

The SMARTER dataset is licensed under the **GNU General Public License v3.0**.
The license is found in the 'LICENSE' file.
