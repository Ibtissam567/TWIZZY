/**
 * 
 */
package com.logiciel;
import com.logiciel.screen.HomeScreen;
import javax.swing.SwingUtilities;

/**
 * 
 */
public class Main {

	/**
	 * 
	 */
	public Main() {
		// TODO Auto-generated constructor stub
	}

	/**
	 * @param args
	 */
	public static void main(String[] args) {
		// TODO Auto-generated method stub
		SwingUtilities.invokeLater(() -> {
			HomeScreen fenetre = new HomeScreen();
			fenetre.setVisible(true);
		});
	}

}
