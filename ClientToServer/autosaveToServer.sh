for var in `seq 1 100`;do 
 echo $var

  sudo ./l2 aTtest$var 
  sshpass -p 1234 scp aTtest$var yhk@10.3.60.107:~/Desktop/csi-motion-main/matlab_converter/
done

