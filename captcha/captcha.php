<?php

// ������ ������ ��������, ������������ � �����
$capletters = 'ABCDEFGKIJKLMNOPQRSTUVWXYZ123456789'; 
// ����� ����� 7 ������
$captlen = 7; 

// ������������� ������� �����������
$capwidth = 120; $capheight = 20; 

// ���������� �����
$capfont = 'E:/GitHub/captcha-recognition/captcha/font/comic.ttf'; 

 // ������ ������ ������
$capfontsize = 14;

// �������������� HTTP ���������, ����� ������� ������ 
// ������� ����������� ����� �� �����, � �����������
header('Content-type: image/png'); 

// ����������� ����������� � ���������� ����� ���������
$capim = imagecreatetruecolor($capwidth, $capheight); 

// ������������� ������������� ���������� ����� ������ (������������)
imagesavealpha($capim, true); 

// ������������� ���� ����, � ����� ������ - ����������
$capbg = imagecolorallocatealpha($capim, 0, 0, 0, 127);

// ������������� ���� ���� �����������
imagefill($capim, 0, 0, $capbg); 

// ������ ��������� �������� �����
$capcha = '';

// ��������� ���� ���������� �����������
for ($i = 0; $i < $captlen; $i++){

// �� ������ ������ ����� ���������� ������ � ��������� � �����
$capcha .= $capletters[rand(0, strlen($capletters)-1) ]; 

// ���������� ��������� ������� �� X ���
$x = ($capwidth - 20) / $captlen * $i + 10;

// ������� ������������ � ��� ���������.
$x = rand($x, $x+5); 

// ������� ��������� �� Y ���
$y = $capheight - ( ($capheight - $capfontsize) / 2 ); 

// ������ ��������� ���� ��� �������.
$capcolor = imagecolorallocate($capim, rand(0, 100), rand(0, 100), rand(0, 100) ); 

// ������ ��� �������
$capangle = rand(-25, 25); 

// ������ ��������� ������, �������� ��� ��������� ���������
imagettftext($capim, $capfontsize, $capangle, $x, $y, $capcolor, $capfont, $capcha[$i]);

} // ��������� ����
$path = 'E:/GitHub/captcha-recognition/captcha/captchas/'.$capcha.'.png';
imagepng($capim, $path);

imagedestroy($capim); // ������� ������.

?> 