from sqlalchemy import create_engine, Column, Integer, String, Sequence, ForeignKey, Text, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker

# Define the SQLite database URL
DATABASE_URL = "sqlite:///blog.db"

# Create the database engine
engine = create_engine(DATABASE_URL)

# Create a declarative base class
Base = declarative_base()

# Define the User model class
class User(Base):
    __tablename__ = 'users'
    id = Column(Integer, Sequence('user_id_seq'), primary_key=True)
    name = Column(String(50), nullable=False)
    email = Column(String(50), unique=True, nullable=False)
    posts = relationship('Post', back_populates='author')

    def __repr__(self):
        return f"<User(name='{self.name}', email='{self.email}')>"

# Define the Post model class
class Post(Base):
    __tablename__ = 'posts'
    id = Column(Integer, Sequence('post_id_seq'), primary_key=True)
    title = Column(String(100), nullable=False)
    content = Column(Text, nullable=False)
    user_id = Column(Integer, ForeignKey('users.id'))
    author = relationship('User', back_populates='posts')
    comments = relationship('Comment', back_populates='post')

    def __repr__(self):
        return f"<Post(title='{self.title}', content='{self.content[:20]}...', author='{self.author.name}')>"

# Define the Comment model class
class Comment(Base):
    __tablename__ = 'comments'
    id = Column(Integer, Sequence('comment_id_seq'), primary_key=True)
    content = Column(Text, nullable=False)
    user_id = Column(Integer, ForeignKey('users.id'))
    post_id = Column(Integer, ForeignKey('posts.id'))
    user = relationship('User')
    post = relationship('Post', back_populates='comments')

    def __repr__(self):
        return f"<Comment(content='{self.content[:20]}...', user='{self.user.name}', post='{self.post.title}')>"

# Create all tables
Base.metadata.create_all(engine)

# Create a sessionmaker
Session = sessionmaker(bind=engine)
session = Session()

# Add sample data
def add_sample_data():
    user1 = User(name='Alice', email='alice@example.com')
    user2 = User(name='Bob', email='bob@example.com')
    post1 = Post(title='First Post', content='This is the first post content', author=user1)
    post2 = Post(title='Second Post', content='This is the second post content', author=user2)
    comment1 = Comment(content='Nice post!', user=user2, post=post1)
    comment2 = Comment(content='Thanks!', user=user1, post=post1)
    comment3 = Comment(content='Interesting read.', user=user1, post=post2)
    session.add_all([user1, user2, post1, post2, comment1, comment2, comment3])
    session.commit()

# Function to fetch data using session
def fetch_data_with_session():
    users = session.query(User).all()
    posts = session.query(Post).all()
    comments = session.query(Comment).all()
    return users, posts, comments

# Function to fetch data using connection
def fetch_data_with_connection():
    with engine.connect() as connection:
        result = connection.execute(text("SELECT * FROM users"))
        users = result.fetchall()
        return users

# Main function to set up the database, add data, and fetch data
if __name__ == "__main__":
    add_sample_data()
    
    # Fetch data using session
    users, posts, comments = fetch_data_with_session()
    print("Data fetched using session:")
    for user in users:
        print(user)
    for post in posts:
        print(post)
    for comment in comments:
        print(comment)
    
    # Fetch data using connection
    users_with_connection = fetch_data_with_connection()
    print("\nData fetched using connection:")
    for user in users_with_connection:
        print(user)
