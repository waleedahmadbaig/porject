## List of important Django commands
Run these command from the terminal location where manage.py is located
- 'python manage.py runserver' - to run the server locally on your system.
- 'python manage.py makemigrations' & 'python manage.py migrate' - Apply the changes to the database (if you make any).
- 'python manage.py dbshell' - To start the sqlite shell and explore the database. And use the following commands in the shell:
    - '.table' - To check out the tables created
    - 'select * from {tablename}' - To check out the data inside the table.
    - 'delete from {tablename}' - Use caution. Deletes the entire table data. Used to clear out the database.