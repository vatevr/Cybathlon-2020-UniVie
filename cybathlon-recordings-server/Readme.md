Running a database

```shell script
docker run --rm --name pg-cybathlon -e POSTGRES_PASSWORD=docker -d -p 5433:5432 -v $HOME/docker/volumes/postgres:/var/lib/postgresql/data postgres
```

### Boot the server instance and database

Create the containers for server and `postgres`

```shell script
docker-compose up -d
```

Migrate the postgres schema if necessary

```shell script
make migrate
```

Drop tables for testing

```shell script
make drop
```