Running a database

```shell script
docker run --rm --name pg-cybathlon -e POSTGRES_PASSWORD=docker -d -p 5433:5432 -v $HOME/docker/volumes/postgres:/var/lib/postgresql/data postgres
```
