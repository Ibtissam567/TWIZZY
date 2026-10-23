package com.logiciel.act;
import java.awt.image.BufferedImage;
import java.io.ByteArrayInputStream;
import java.io.File;
import java.io.InputStream;
import java.util.Vector;

import javax.imageio.ImageIO;
import javax.swing.ImageIcon;
import javax.swing.JFrame;
import javax.swing.JLabel;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.highgui.Highgui;

public class Activite2 {

	public static void main(String[] args) {
		// TODO Auto-generated method stubCE
		System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
		//Lecture du fichier image et transformation en matrice
		Mat m=LectureImage("Images/bgr.png");
		//Cr�ation d'un vecteur de matrices
		Vector<Mat> channels=new Vector<Mat>();
		//d�coupage de la matrice en 3 canneaux BGR
		Core.split(m,channels);
		//for (int i=0;i<channels.size();i++){
		//	afficheImage("Voie n : "+Integer.toString(i),channels.get(i));
		//}
		//recupeartion du canal Green d'indice 1
		afficheImage("Voie verte : ",channels.get(1));
		
	}


	//Methode qui ouvre un fichier et renvoie la matrice au format OpenCV (la meme que dans l'activite 0)
	public static Mat LectureImage(String fichier){
		File f=new File(fichier);
		Mat m=Highgui.imread(f.getAbsolutePath());
		return m;

	}
	//Methode qui affiche une matrice dans une fenetre Java (inutile d'analyser ce code dans le cadre de cette �preuve)
	public static void afficheImage(String title, Mat img){
		MatOfByte matOfByte=new MatOfByte();
		Highgui.imencode(".png",img,matOfByte);
		byte[] byteArray=matOfByte.toArray();
		BufferedImage bufImage=null;
		try{
			InputStream in=new ByteArrayInputStream(byteArray);
			bufImage=ImageIO.read(in);
			JFrame frame=new JFrame();
			frame.setTitle(title);
			frame.getContentPane().add(new JLabel(new ImageIcon(bufImage)));
			frame.pack();
			frame.setVisible(true);

		}
		catch(Exception e){
			e.printStackTrace();
		}


	}

}
