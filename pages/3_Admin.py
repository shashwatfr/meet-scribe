"""Admin page: row counts, database size and every table, with password hashes hidden.

Calls services/admin.py, which checks the admin role in the database itself.
Normal users only see "Admins only."
"""
import streamlit as st

from services import admin
from ui import require_login

user = require_login()
st.title("Admin")

try:
    admin.require_admin(user["id"])  # role read from the database, not the session
except PermissionError:
    st.error("Admins only.")
    st.stop()

sizes = admin.db_file_size(user["id"])
counts = admin.row_counts(user["id"])
columns = st.columns(len(counts) + 1)
for column, (table, count) in zip(columns, counts.items()):
    column.metric(table, count)
columns[-1].metric("database file", f"{(sizes['database'] + sizes['wal']) / 1024:,.0f} KB")
st.caption(f"Main file {sizes['database']:,} bytes, WAL file {sizes['wal']:,} bytes. "
           "Tables are shown as stored, with long values cut to 200 characters. "
           "Password hashes are replaced by (hidden).")

for table in admin.TABLE_QUERIES:
    st.subheader(table)
    rows = admin.table_rows(user["id"], table)
    if rows:
        st.dataframe(rows, hide_index=True)
    else:
        st.caption("No rows.")
