const SERVER_URL = 'http://' + window.location.host;
const SEND_BUTTON = document.getElementById('SendButton');
const INPUT = document.getElementById('InputFrame_Text');
const SHOW_DIV = document.getElementById('ShowFrame');
const SCROLL_FRAME = document.getElementById('ScrollFrame');
var user_id = undefined;
// 显示逻辑
function getTime(){
    let time=new Date();
    let year=time.getFullYear(),
        month=time.getMonth()+1,
        day=time.getDate(),
        h=time.getHours(),
        m=time.getMinutes(),
        s=time.getSeconds();
    return year+"."+(month<10?"0":"")+month+"."+(day<10?"0":"")+day+";"+
            (h<10?"0":"")+h+":"+(m<10?"0":"")+m+":"+(s<10?"0":"")+s;
}
function show_msg(name, content){
    if(content=='') return;
    var nameNode = document.createElement('div');
    nameNode.innerHTML = name;
    nameNode.className = 'UserName';
    var timeNode = document.createElement('div');
    timeNode.innerHTML = getTime();
    timeNode.className = 'time';
    var contentNode = document.createElement('div');
    contentNode.innerHTML = content;
    contentNode.className = 'UserMsg';
    var msgNode = document.createElement('div');
    msgNode.append(nameNode);
    msgNode.append(timeNode);
    msgNode.append(contentNode);
    SHOW_DIV.appendChild(msgNode);
    SCROLL_FRAME.scrollTop = SCROLL_FRAME.scrollHeight;
}
// 交流逻辑
SEND_BUTTON.onclick = ()=>{
    // 内容
    var user_content = INPUT.value;
    INPUT.value = '';
    if(user_content=='\\quit'){
        var httpRequest = new XMLHttpRequest();
        httpRequest.open('POST', SERVER_URL, true);
        httpRequest.setRequestHeader("Content-type", "text/plain");
        httpRequest.setRequestHeader("Content-length", user_content.length);
        httpRequest.send('\\quit');
        httpRequest.onreadystatechange = function () {window.close();};
    }
    console.log(user_content);
    // 连接服务器
    var httpRequest = new XMLHttpRequest();
    httpRequest.open('POST', SERVER_URL, true);
    httpRequest.setRequestHeader("Content-type", "text/plain");
    httpRequest.setRequestHeader("Content-length", user_content.length);
    httpRequest.send(user_content);
    show_msg('user', user_content);
    // 回调显示
    httpRequest.onreadystatechange = function () {
        // 回调
        if (httpRequest.readyState == 4 && httpRequest.status == 200) {
            var content = httpRequest.responseText;
            console.log(content);
            show_msg('assistant', content);
        }
    };
}
// 离开quit
window.addEventListener('beforeunload', (ev)=>{
    ev.preventDefault();
    var httpRequest = new XMLHttpRequest();
    httpRequest.open('POST', SERVER_URL, true);
    httpRequest.setRequestHeader("Content-type", "text/plain");
    httpRequest.setRequestHeader("Content-length", user_content.length);
    httpRequest.send('\\quit');
    httpRequest.onreadystatechange = function () {window.close();};
    alert('您真的要离开？');
    return 'really want to quit?'
})
