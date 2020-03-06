DBPORT=5432

migrate:
	@echo $(PYTHONPATH)
	PYTHONPATH=. python3 ./src/tools/migrate.py migrate

drop:
	@echo $(PYTHONPATH)
	PYTHONPATH=. python3 ./src/tools/migrate.py drop
