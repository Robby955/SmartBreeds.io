document.addEventListener('DOMContentLoaded', function() {
    const userList = document.getElementById('user-list');

    // Fetch users from the backend
    fetch('/api/users')
        .then(response => response.json())
        .then(users => {
            // Iterate over the user data and append rows to the table
            users.forEach(user => {
                const userRow = document.createElement('tr');
                userRow.innerHTML = `<td>${user.id}</td><td>${user.username}</td><td>${user.email}</td>`;
                userList.appendChild(userRow);
            });
        })
        .catch(error => {
            console.error('Error fetching user data:', error);
            // Handle errors or display a message to the user
        });
});
