const express = require('express');
const path = require('path');

const app = express();
const port = 3000;

app.use(express.static('static'));
app.get('/', (req, res)=>{
    res.sendFile('./templates/index.html', {root:__dirname});
})

app.listen(port, ()=>{
    console.log(`Example app listening on port ${port}`);
})

app.use(function(err, req, res, next){
    console.log(err.stack);
    res.status(500);
})