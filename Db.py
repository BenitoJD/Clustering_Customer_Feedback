from peewee import *

def setup_database():
    db = SqliteDatabase('clustersDB.db')

    class BaseModel(Model):
        class Meta:
            database = db

    class Cluster(BaseModel):
        cluster_id = AutoField(primary_key=True)
        name = TextField(null=False)                 

    class Item(BaseModel):
        item_id = AutoField(primary_key=True)     
        description = TextField()                    
        cluster_id = ForeignKeyField(Cluster, backref='items', on_delete='CASCADE')

    db.connect()
    db.create_tables([Cluster, Item], safe = True)

    cluster = Cluster.create(name="Sample Cluster")
    Item.create(description="Item 1 description", cluster_id=cluster)
    Item.create(description="Item 2 description", cluster_id=cluster)

    db.close()

setup_database()
