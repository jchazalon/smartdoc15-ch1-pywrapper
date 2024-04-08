# docker automation
If you just need to download data and don't want to deal with proper python and other dependencies installation, then you can use docker to automate it.

install docker and docker-compose

Run in docker-compose `docker-compose.yml` file inside of `docker-automation` folder like

`docker-compose up`

It will install required python versions alongside with all required packages and dependencies to build these packages into container.

Then it will run `docker-entrypoint.py` file. 
There you just define a dataset and a place to download files.