<?php
  //--��������� ���� ����� ���� �������
  if ( isset($_POST['capcha']) )
  {
    $code = $_POST['capcha']; //�������� ���������� �����
    session_start();

    if ( isset($_SESSION['capcha']) && strtoupper($_SESSION['capcha']) == strtoupper($code) ) //���������� ��������� ����� � ����������� � ���������� � ������
      echo '�����!';
    else
      echo '�� �����!';
    //������� ����� �� ������ 
    unset($_SESSION['capcha']);
    exit();
  }
?>

<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.01 Transitional//EN" "http://www.w3.org/TR/html4/loose.dtd">
<html>
  <head>
    <title>����</title>
    <meta http-equiv="Content-Type" content="text/html; charset=windows-1251">
  </head>
  <body>
    <form method="post" action="<?php echo $_SERVER['PHP_SELF']; ?>">
      <img src="captcha.php" width="120" height="20"><br>
      <input type="text" name="capcha"><br>
      <input type="submit" value="���������">
    </form>
  </body> 
</html>

