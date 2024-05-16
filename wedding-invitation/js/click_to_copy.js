function copy(that){
var inp =document.createElement('input');
document.body.appendChild(inp);
inp.value =that.textContent.split(' ')[2];
inp.select();
document.execCommand('copy',false);
inp.remove();
  alert("계좌번호가 복사되었습니다.");
}
