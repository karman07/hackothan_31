from app.db import db

async def user_exists(candidate_name: str, parent_name: str, institute: str) -> bool:
    query = {
        "candidate_name": {"$regex": candidate_name, "$options": "i"},
        "parent_name": {"$regex": parent_name, "$options": "i"},
        "institute": {"$regex": institute, "$options": "i"},
    }
    print(query)
    user = await db.users.find_one(query)
    print(db, db.users, user)
    return user is not None
