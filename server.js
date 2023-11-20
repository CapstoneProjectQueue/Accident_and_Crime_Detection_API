const axios = require('axios');
const express = require('express');
const router = express.Router();
const bodyParser = require("body-parser");

const app = express();
const port = 3000;

app.use(express.static('static'));
app.use(bodyParser.json());
app.use(bodyParser.urlencoded({extended:true}));
app.get('/', (req, res)=>{
    res.sendFile('./templates/index.html', {root:__dirname});
})
router.post('/', (req, res)=>{
    let body = req.body;
    let file_name = body.file_name;

    const fileRequest = (callback)=>{
        const options = {
            method:'POST',
            uri:"http://127.0.0.1:5000/predict",
            qs:{
                file_name:file_name
            }
        }
        request(options, (err, res, body)=>{
            callbackify(undefined, {
                result:body
            });
        });
    }
})
app.listen(port, ()=>{
    console.log(`Example app listening on port ${port}`);
})

app.use(function(err, req, res, next){
    console.log(err.stack);
    res.status(500);
})

module.exports = router;